import asyncio
import websockets
import json
import base64
import pyaudio
import os
import struct

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("A variável GEMINI_API_KEY não foi configurada!")

MODEL_NAME = "gemini-3.8-live"
WS_URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}"

# Configurações de Áudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000      # 16kHz Entrada
OUTPUT_RATE = 24000     # 24kHz Saída
CHUNK_SIZE = 1024

# Sensibilidade do Microfone (Ajuste se necessário)
SILENCE_THRESHOLD = 600   # Volume mínimo para considerar fala (picos de 0 a 32767)
SILENCE_SECONDS = 1.0     # Quantos segundos de silêncio acionam a resposta do Gemini

p = pyaudio.PyAudio()


# 1. GRAVAÇÃO E ENVIO COM DETECÇÃO DE SILÊNCIO
async def send_mic_audio(websocket):
    mic_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=INPUT_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("\n🎤 Microfone ativo! Pode falar...")

    user_is_speaking = False
    silence_chunks = 0

    max_silence_chunks = int(
        (INPUT_RATE / CHUNK_SIZE) * SILENCE_SECONDS
    )

    try:
        while True:

            data = await asyncio.to_thread(
                mic_stream.read,
                CHUNK_SIZE,
                exception_on_overflow=False
            )

            samples = struct.unpack(
                "<" + "h" * (len(data) // 2),
                data
            )

            peak = max(abs(x) for x in samples)

            # ==========================================
            # USUÁRIO COMEÇOU A FALAR
            # ==========================================

            if peak > SILENCE_THRESHOLD:

                if not user_is_speaking:
                    print("🗣️ Voz detectada...")
                    user_is_speaking = True

                silence_chunks = 0

            # ==========================================
            # SE ESTÁ FALANDO, ENVIA ÁUDIO
            # ==========================================

            if user_is_speaking:

                encoded_data = base64.b64encode(
                    data
                ).decode("utf-8")

                audio_message = {
                    "realtimeInput": {
                        "audio": {
                            "data": encoded_data,
                            "mimeType": "audio/pcm;rate=16000"
                        }
                    }
                }

                await websocket.send(
                    json.dumps(audio_message)
                )

                # --------------------------------------
                # SILÊNCIO DURANTE A FALA
                # --------------------------------------

                if peak <= SILENCE_THRESHOLD:
                    silence_chunks += 1

                # --------------------------------------
                # FIM DA FALA
                # --------------------------------------

                if silence_chunks >= max_silence_chunks:

                    print(
                        "⏳ Fim da fala → "
                        "audioStreamEnd"
                    )

                    await websocket.send(
                        json.dumps({
                            "realtimeInput": {
                                "audioStreamEnd": True
                            }
                        })
                    )

                    user_is_speaking = False
                    silence_chunks = 0

            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        pass

    finally:
        mic_stream.stop_stream()
        mic_stream.close()
# 2. RECEBIMENTO E REPRODUÇÃO
async def receive_and_play_audio(websocket):
    speaker_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=OUTPUT_RATE,
        output=True,
        frames_per_buffer=1024
    )

    try:
        async for message in websocket:
            response = json.loads(message)

            if "error" in response:
                print(f"❌ Erro do Gemini: {response['error']}")
                break

            server_content = response.get("serverContent", {})
            model_turn = server_content.get("modelTurn", {})
            parts = model_turn.get("parts", [])

            for part in parts:
                inline_data = part.get("inlineData", {})

                if inline_data.get("mimeType", "").startswith("audio/pcm"):
                    audio_bytes = base64.b64decode(inline_data["data"])
                    await asyncio.to_thread(speaker_stream.write, audio_bytes)

            if server_content.get("turnComplete"):
                print("✅ Gemini terminou de responder. Pode falar novamente!\n")

    except asyncio.CancelledError:
        pass
    finally:
        speaker_stream.stop_stream()
        speaker_stream.close()


# 3. MAIN
async def main():
    async with websockets.connect(WS_URL) as websocket:
        print("🔗 Conectado ao Gemini Live!")

        # Handshake de Setup com instrução em Português
        setup_message = {
            "setup": {
                "model": f"models/{MODEL_NAME}",
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
        }
        await websocket.send(json.dumps(setup_message))

        # Aguarda confirmação do Gemini
        first_msg = await websocket.recv()
        ack = json.loads(first_msg)

        if "setupComplete" in ack:
            print("✅ Setup concluído com sucesso!")
        else:
            print("⚠️ Setup retornou:", ack)

        # Executa envio e recebimento concorrentemente
        send_task = asyncio.create_task(send_mic_audio(websocket))
        recv_task = asyncio.create_task(receive_and_play_audio(websocket))

        await asyncio.gather(send_task, recv_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Encerrado pelo usuário.")
    finally:
        p.terminate()
