use base64::{engine::general_purpose, Engine as _};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;

const MODEL_NAME: &str = "gemini-2.0-flash-exp"; // Modelo multimodal live recomendado
const INPUT_RATE: u32 = 16000;
const OUTPUT_RATE: u32 = 24000;
const SILENCE_THRESHOLD: i16 = 600;
const SILENCE_SECONDS: f32 = 1.0;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Carrega o arquivo .env
    dotenvy::dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| "A variável GEMINI_API_KEY não foi configurada no .env!")?;

    let ws_url = format!(
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={}",
        api_key
    );

    println!("🔗 Conectando ao Gemini Live...");
    let (ws_stream, _) = connect_async(&ws_url).await?;
    let (mut ws_write, mut ws_read) = ws_stream.split();
    println!("✅ Conectado com sucesso!");

    // Handshake de Setup
    let setup_message = json!({
        "setup": {
            "model": format!("models/{}", MODEL_NAME),
            "generationConfig": {
                "responseModalities": ["AUDIO"]
            },
            "systemInstruction": {
                "parts": [
                    {
                        "text": "Você é um assistente conversacional em tempo real. Responda SEMPRE em Português do Brasil (pt-BR). Use frases curtas, tom natural, direto e expressivo. Seu nome será Otto e você é de um projeto de escola."
                    }
                ]
            }
        }
    });

    ws_write.send(tokio_tungstenite::tungstenite::Message::Text(setup_message.to_string())).await?;

    // Aguarda confirmação do Setup
    if let Some(Ok(msg)) = ws_read.next().await {
        if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
            let ack: serde_json::Value = serde_json::from_str(&text)?;
            if ack.get("setupComplete").is_some() {
                println!("✅ Setup concluído com sucesso!");
            } else {
                println!("⚠️ Setup retornou: {}", ack);
            }
        }
    }

    // Canais para comunicação entre as threads de áudio e o tokio async
    let (mic_tx, mut mic_rx) = mpsc::channel::<Vec<u8>>(100);
    let (speaker_tx, mut speaker_rx) = mpsc::channel::<Vec<u8>>(100);

    // Inicializa o host de áudio CPAL
    let host = cpal::default_host();

    // ---------------------------------------------------------
    // CONFIGURAÇÃO DO MICROFONE (INPUT)
    // ---------------------------------------------------------
    let input_device = host.default_input_device().ok_or("Nenhum microfone encontrado")?;
    let mut input_config: cpal::StreamConfig = input_device.default_input_config()?.into();
    input_config.sample_rate = cpal::SampleRate(INPUT_RATE);
    input_config.channels = 1;

    let mic_tx_clone = mic_tx.clone();
    let input_stream = input_device.build_input_stream(
        &input_config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            let _ = mic_tx_clone.blocking_send(data.iter().flat_map(|v| v.to_le_bytes()).collect());
        },
        |err| eprintln!("❌ Erro no microfone: {}", err),
        None,
    )?;
    input_stream.play()?;

    // ---------------------------------------------------------
    // CONFIGURAÇÃO DO AUTORAL / CAIXA DE SOM (OUTPUT)
    // ---------------------------------------------------------
    let output_device = host.default_output_device().ok_or("Nenhum alto-falante encontrado")?;
    let mut output_config: cpal::StreamConfig = output_device.default_output_config()?.into();
    output_config.sample_rate = cpal::SampleRate(OUTPUT_RATE);
    output_config.channels = 1;

    // Fila thread-safe para o fluxo de saída de áudio
    let audio_buffer = Arc::new(tokio::sync::Mutex  ::new(std::collections::VecDeque::new()));
    let buffer_clone = Arc::clone(&audio_buffer);

    let output_stream = output_device.build_output_stream(
        &output_config,
        move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
            let mut buf = buffer_clone.blocking_lock();
            for sample in data.iter_mut() {
                if let Some(val) = buf.pop_front() {
                    *sample = val;
                } else {
                    *sample = 0;
                }
            }
        },
        |err| eprintln!("❌ Erro no alto-falante: {}", err),
        None,
    )?;
    output_stream.play()?;

    // Task para consumir o buffer e alimentar o alto-falante assincronamente
    let buffer_clone2 = Arc::clone(&audio_buffer);
    tokio::spawn(async move {
        while let Some(bytes) = speaker_rx.recv().await {
            let samples: Vec<i16> = bytes
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            let mut buf = buffer_clone2.lock().await;
            for s in samples {
                buf.push_back(s);
            }
        }
    });

    println!("\n🎤 Microfone ativo! Pode falar...\n");

    // ---------------------------------------------------------
    // LOOP DE ENVIO (VAD / SILENCE DETECTION)
    // ---------------------------------------------------------
    let ws_write_task = tokio::spawn(async move {
        let mut user_is_speaking = false;
        let mut silence_chunks = 0;
        let max_silence_chunks = ((INPUT_RATE as f32 / 1024.0) * SILENCE_SECONDS) as i32;

        while let Some(data) = mic_rx.recv().await {
            let samples: Vec<i16> = data
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();

            let peak = samples.iter().map(|x| x.abs()).max().unwrap_or(0);

            if peak > SILENCE_THRESHOLD {
                if !user_is_speaking {
                    println!("🗣️ Voz detectada...");
                    user_is_speaking = true;
                }
                silence_chunks = 0;
            }

            if user_is_speaking {
                let encoded_data = general_purpose::STANDARD.encode(&data);
                let audio_message = json!({
                    "realtimeInput": {
                        "audio": {
                            "data": encoded_data,
                            "mimeType": "audio/pcm;rate=16000"
                        }
                    }
                });

                if ws_write.send(tokio_tungstenite::tungstenite::Message::Text(audio_message.to_string())).await.is_err() {
                    break;
                }

                if peak <= SILENCE_THRESHOLD {
                    silence_chunks += 1;
                }

                if silence_chunks >= max_silence_chunks {
                    println!("⏳ Fim da fala → audioStreamEnd");
                    let end_message = json!({
                        "realtimeInput": {
                            "audioStreamEnd": true
                        }
                    });
                    let _ = ws_write.send(tokio_tungstenite::tungstenite::Message::Text(end_message.to_string())).await;

                    user_is_speaking = false;
                    silence_chunks = 0;
                }
            }
        }
    });

    // ---------------------------------------------------------
    // LOOP DE RECEBIMENTO
    // ---------------------------------------------------------
    let ws_read_task = tokio::spawn(async move {
        while let Some(Ok(message)) = ws_read.next().await {
            if let tokio_tungstenite::tungstenite::Message::Text(text) = message {
                if let Ok(response) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(err) = response.get("error") {
                        println!("❌ Erro do Gemini: {}", err);
                        break;
                    }

                    let server_content = response.get("serverContent");
                    if let Some(content) = server_content {
                        if let Some(model_turn) = content.get("modelTurn") {
                            if let Some(parts) = model_turn.get("parts").and_then(|p| p.as_array()) {
                                for part in parts {
                                    if let Some(inline_data) = part.get("inlineData") {
                                        if let Some(mime) = inline_data.get("mimeType").and_then(|m| m.as_str()) {
                                            if mime.starts_with("audio/pcm") {
                                                if let Some(base64_data) = inline_data.get("data").and_then(|d| d.as_str()) {
                                                    if let Ok(decoded_bytes) = general_purpose::STANDARD.decode(base64_data) {
                                                        let _ = speaker_tx.send(decoded_bytes).await;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if content.get("turnComplete").and_then(|t| t.as_bool()).unwrap_or(false) {
                            println!("✅ Gemini terminou de responder. Pode falar novamente!\n");
                        }
                    }
                }
            }
        }
    });

    tokio::select! {
        _ = ws_write_task => {},
        _ = ws_read_task => {},
    }

    Ok(())
}
