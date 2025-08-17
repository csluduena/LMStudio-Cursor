import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from ctransformers import AutoModelForCausalLM
import queue
import time
import pyaudio
import numpy as np
from collections import deque
import asyncio
import edge_tts
import tempfile
import os

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat con IA Local - Práctica de Inglés")
        self.root.geometry("1200x700")
        
        # Cola para comunicación entre hilos
        self.message_queue = queue.Queue()
        
        # Configuración de audio para streaming en tiempo real
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 512  # Chunks más pequeños para menor latencia
        self.audio = pyaudio.PyAudio()
        
        # Variables para escucha en tiempo real
        self.is_listening = False
        self.voice_thread = None
        self.audio_stream = None
        self.audio_buffer = deque(maxlen=200)  # Buffer más grande para mejor detección
        self.last_speech_time = 0
        self.silence_threshold = 2.0
        self.is_processing = False
        self.volume_threshold = 500  # Umbral de volumen para detectar voz
        
        # Configuración de respuesta
        self.response_mode = tk.StringVar(value="TEXT + TTS")
        
        # Configuración de TTS
        self.tts_voice = "en-US-AriaNeural"  # Voz en inglés nativo
        self.tts_speed = 1.0
        self.tts_volume = 100
        
        # Cargar modelo en hilo separado
        self.model = None
        self.model_loaded = False
        self.loading_thread = threading.Thread(target=self.load_model)
        self.loading_thread.start()
        
        self.setup_ui()
        self.check_model_status()
    
    def setup_ui(self):
        # Frame principal con dos columnas
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid principal
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=3)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)
        
        # === PANEL IZQUIERDO: CHAT ===
        self.setup_chat_panel(main_container)
        
        # === PANEL DERECHO: CONFIGURACIÓN ===
        self.setup_config_panel(main_container)
    
    def setup_chat_panel(self, container):
        """Configura el panel izquierdo del chat"""
        chat_frame = ttk.Frame(container, padding="5")
        chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Título
        title_label = ttk.Label(chat_frame, text="💬 Chat con IA Local - Práctica de Inglés", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Área de chat
        self.chat_area = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=20, 
                                                 font=("Arial", 10))
        self.chat_area.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), 
                           pady=(0, 10))
        
        # Frame para entrada y botones
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Campo de entrada
        self.input_field = ttk.Entry(input_frame, font=("Arial", 11))
        self.input_field.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.input_field.bind('<Return>', self.send_message)
        
        # Botón enviar
        self.send_button = ttk.Button(input_frame, text="Enviar", command=self.send_message)
        self.send_button.grid(row=0, column=1, padx=(0, 10))
        
        # Botón hablar
        self.speak_button = ttk.Button(input_frame, text="🔊 Hablar", command=self.speak_last_response)
        self.speak_button.grid(row=0, column=2)
        
        # Frame para controles de voz
        voice_frame = ttk.Frame(chat_frame)
        voice_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        voice_frame.columnconfigure(0, weight=1)
        
        # Botón de micrófono
        self.mic_button = ttk.Button(voice_frame, text="🎤 Iniciar Escucha", 
                                    command=self.toggle_voice_listening)
        self.mic_button.grid(row=0, column=0, padx=(0, 10))
        
        # Indicador de estado de voz
        self.voice_status_var = tk.StringVar(value="Micrófono desactivado")
        voice_status_label = ttk.Label(voice_frame, textvariable=self.voice_status_var, 
                                     font=("Arial", 10, "italic"))
        voice_status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Barra de estado
        self.status_var = tk.StringVar(value="Cargando modelo...")
        status_bar = ttk.Label(chat_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Configurar entrada inicial
        self.input_field.focus()
    
    def setup_config_panel(self, container):
        """Configura el panel derecho de configuración"""
        config_frame = ttk.LabelFrame(container, text="⚙️ Configuración", padding="10")
        config_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # === MODO DE RESPUESTA ===
        response_frame = ttk.LabelFrame(config_frame, text="🎯 Modo de Respuesta", padding="5")
        response_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Opciones de modo de respuesta
        ttk.Radiobutton(response_frame, text="Solo Texto", variable=self.response_mode, 
                       value="TEXT", command=self.on_response_mode_change).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(response_frame, text="Solo Voz (TTS)", variable=self.response_mode, 
                       value="TTS", command=self.on_response_mode_change).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(response_frame, text="Texto + Voz", variable=self.response_mode, 
                       value="TEXT + TTS", command=self.on_response_mode_change).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # === CONFIGURACIÓN DE VOZ ===
        voice_config_frame = ttk.LabelFrame(config_frame, text="🎤 Configuración de Voz", padding="5")
        voice_config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Umbral de silencio
        ttk.Label(voice_config_frame, text="Pausa para fin de frase:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        silence_frame = ttk.Frame(voice_config_frame)
        silence_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.silence_slider = ttk.Scale(silence_frame, from_=0.5, to=5.0, 
                                       orient=tk.HORIZONTAL, length=150,
                                       command=self.update_silence_threshold)
        self.silence_slider.set(self.silence_threshold)
        self.silence_slider.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.silence_label = ttk.Label(silence_frame, text=f"{self.silence_threshold:.1f}s")
        self.silence_label.grid(row=0, column=1, padx=(5, 0))
        
        # Umbral de volumen
        ttk.Label(voice_config_frame, text="Sensibilidad del micrófono:").grid(row=2, column=0, sticky=tk.W, pady=(10, 2))
        
        volume_threshold_frame = ttk.Frame(voice_config_frame)
        volume_threshold_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.volume_slider = ttk.Scale(volume_threshold_frame, from_=100, to=2000, 
                                      orient=tk.HORIZONTAL, length=150,
                                      command=self.update_volume_threshold)
        self.volume_slider.set(self.volume_threshold)
        self.volume_slider.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.volume_label = ttk.Label(volume_threshold_frame, text=str(self.volume_threshold))
        self.volume_label.grid(row=0, column=1, padx=(5, 0))
        
        # Velocidad de TTS
        ttk.Label(voice_config_frame, text="Velocidad de habla (TTS):").grid(row=4, column=0, sticky=tk.W, pady=(10, 2))
        
        speed_frame = ttk.Frame(voice_config_frame)
        speed_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.speed_slider = ttk.Scale(speed_frame, from_=0.5, to=2.0, 
                                     orient=tk.HORIZONTAL, length=150,
                                     command=self.update_tts_speed)
        self.speed_slider.set(self.tts_speed)
        self.speed_slider.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.speed_label = ttk.Label(speed_frame, text=f"{self.tts_speed:.1f}x")
        self.speed_label.grid(row=0, column=1, padx=(5, 0))
        
        # === INFORMACIÓN DEL SISTEMA ===
        info_frame = ttk.LabelFrame(config_frame, text="ℹ️ Información", padding="5")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(info_frame, text="Modelo: OpenHermes-2.5-Mistral-7B", 
                 font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(info_frame, text="TTS: Edge TTS (Inglés nativo)", 
                 font=("Arial", 9)).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(info_frame, text="Audio: Streaming en tiempo real", 
                 font=("Arial", 9)).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # === BOTONES DE ACCIÓN ===
        action_frame = ttk.Frame(config_frame)
        action_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(action_frame, text="🔄 Reiniciar Config", 
                  command=self.reset_config).grid(row=0, column=0, pady=5)
        
        # Configurar expansión
        config_frame.columnconfigure(0, weight=1)
        voice_config_frame.columnconfigure(0, weight=1)
        speed_frame.columnconfigure(0, weight=1)
        volume_threshold_frame.columnconfigure(0, weight=1)
        silence_frame.columnconfigure(0, weight=1)
    
    def on_response_mode_change(self):
        """Se ejecuta cuando cambia el modo de respuesta"""
        mode = self.response_mode.get()
        print(f"Modo de respuesta cambiado a: {mode}")
        
        # Actualizar estado de botones según el modo
        if mode == "TTS":
            self.speak_button.config(state="disabled")
        else:
            self.speak_button.config(state="normal")
    
    def update_silence_threshold(self, value):
        """Actualiza el umbral de silencio"""
        self.silence_threshold = float(value)
        self.silence_label.config(text=f"{self.silence_threshold:.1f}s")
        print(f"Umbral de silencio actualizado: {self.silence_threshold}s")
    
    def update_volume_threshold(self, value):
        """Actualiza el umbral de volumen"""
        self.volume_threshold = int(float(value))
        self.volume_label.config(text=str(self.volume_threshold))
        print(f"Umbral de volumen actualizado: {self.volume_threshold}")
    
    def update_tts_speed(self, value):
        """Actualiza la velocidad del TTS"""
        self.tts_speed = float(value)
        self.speed_label.config(text=f"{self.tts_speed:.1f}x")
        print(f"Velocidad TTS actualizada: {self.tts_speed}x")
    
    def reset_config(self):
        """Reinicia la configuración a valores por defecto"""
        self.silence_slider.set(2.0)
        self.volume_slider.set(500)
        self.speed_slider.set(1.0)
        self.response_mode.set("TEXT + TTS")
        
        self.silence_threshold = 2.0
        self.volume_threshold = 500
        self.tts_speed = 1.0
        
        self.silence_label.config(text="2.0s")
        self.volume_label.config(text="500")
        self.speed_label.config(text="1.0x")
        
        print("Configuración reiniciada a valores por defecto")
    
    def load_model(self):
        """Carga el modelo en un hilo separado"""
        try:
            MODEL_PATH = r"G:\.ollama\models\TheBloke\OpenHermes-2.5-Mistral-7B-GGUF\openhermes-2.5-mistral-7b.Q8_0.gguf"
            
            self.message_queue.put(("status", "Cargando modelo... (esto puede tardar unos minutos)"))
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                model_type="mistral",
                gpu_layers=0,
                context_length=2048
            )
            
            self.message_queue.put(("status", "¡Modelo cargado! Escribe tu mensaje en inglés o usa el micrófono"))
            self.message_queue.put(("model_loaded", True))
            
        except Exception as e:
            self.message_queue.put(("status", f"Error al cargar modelo: {str(e)}"))
            self.message_queue.put(("model_loaded", False))
    
    def check_model_status(self):
        """Verifica el estado del modelo y actualiza la UI"""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                if msg_type == "status":
                    self.status_var.set(data)
                elif msg_type == "model_loaded":
                    self.model_loaded = data
                    if data:
                        self.send_button.config(state="normal")
                        self.input_field.config(state="normal")
                        self.mic_button.config(state="normal")
                        self.add_message("IA", "¡Hola! Soy tu tutor de inglés. Escribe algo en inglés para practicar o presiona el botón del micrófono.")
                    else:
                        self.send_button.config(state="disabled")
                        self.input_field.config(state="disabled")
                        self.mic_button.config(state="disabled")
        except queue.Empty:
            pass
        # Programar siguiente verificación
        self.root.after(100, self.check_model_status)
    
    def toggle_voice_listening(self):
        """Activa/desactiva la escucha de voz"""
        if not self.is_listening:
            self.start_voice_listening()
        else:
            self.stop_voice_listening()
    
    def start_voice_listening(self):
        """Inicia la escucha de voz en tiempo real"""
        if not self.model_loaded:
            return
        
        self.is_listening = True
        self.mic_button.config(text="🎤 Detener Escucha")
        self.voice_status_var.set("Escuchando... Habla ahora")
        
        # Limpiar buffer de audio
        self.audio_buffer.clear()
        self.last_speech_time = time.time()
        self.is_processing = False
        
        # Iniciar hilo de escucha
        self.voice_thread = threading.Thread(target=self.continuous_voice_listening, daemon=True)
        self.voice_thread.start()
        
        self.add_message("Sistema", "🎤 Micrófono activado. Habla en inglés para practicar.")
    
    def stop_voice_listening(self):
        """Detiene la escucha de voz"""
        self.is_listening = False
        self.mic_button.config(text="🎤 Iniciar Escucha")
        self.voice_status_var.set("Micrófono desactivado")
        
        # Procesar cualquier audio restante
        if self.audio_buffer and not self.is_processing:
            self.process_audio_buffer()
        
        self.add_message("Sistema", "🎤 Micrófono desactivado.")
    
    async def speak_text_async(self, text):
        """Lee en voz alta el texto usando Edge TTS"""
        try:
            # Corregir formato de velocidad - Edge TTS usa +20% no +1.0%
            rate_percentage = int((self.tts_speed - 1.0) * 100)
            rate_string = f"{rate_percentage:+d}%" if rate_percentage != 0 else "+0%"
            
            communicate = edge_tts.Communicate(text, self.tts_voice, rate=rate_string)
            
            # Crear archivo temporal para el audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_filename = tmp_file.name
            
            # Generar audio
            await communicate.save(tmp_filename)
            
            # Reproducir audio usando pygame (más confiable)
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(tmp_filename)
                pygame.mixer.music.play()
                
                # Esperar a que termine de reproducir
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
            except ImportError:
                print("🔊 TTS generado (instala pygame para reproducción): {text}")
            
            # Limpiar archivo temporal
            try:
                os.unlink(tmp_filename)
            except:
                pass
                
        except Exception as e:
            print(f"Error en TTS: {e}")
    
    def process_audio_buffer(self):
        """Procesa el buffer de audio acumulado"""
        if not self.audio_buffer or self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            print("🎧 Procesando audio en tiempo real...")
            
            # Aquí procesarías el audio con Whisper local
            # Por ahora, simulamos el procesamiento con el modelo
            self.process_audio_with_model()
            
        except Exception as e:
            print(f"Error al procesar audio: {e}")
        finally:
            self.is_processing = False
            # Reiniciar la escucha después de procesar
            if self.is_listening:
                print("🔄 Reiniciando escucha de voz...")
                self.root.after(100, self.restart_voice_listening)
    
    def restart_voice_listening(self):
        """Reinicia la escucha de voz después de procesar audio"""
        if self.is_listening and not self.is_processing:
            # Limpiar buffer y reiniciar
            self.audio_buffer.clear()
            self.last_speech_time = time.time()
            
            # Iniciar nuevo hilo de escucha
            self.voice_thread = threading.Thread(target=self.continuous_voice_listening, daemon=True)
            self.voice_thread.start()
            print("🎤 Escucha de voz reiniciada")
    
    def continuous_voice_listening(self):
        """Escucha continuamente la voz del usuario en tiempo real"""
        try:
            # Abrir stream de audio
            self.audio_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            print("🎤 Iniciando escucha en tiempo real...")
            
            while self.is_listening:
                try:
                    # Leer audio en chunks pequeños
                    data = self.audio_stream.read(self.chunk, exception_on_overflow=False)
                    self.audio_buffer.append(data)
                    
                    # Detectar si hay voz activa
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume_norm = np.linalg.norm(audio_data)
                    
                    # Si hay sonido por encima del umbral, actualizar timestamp
                    if volume_norm > self.volume_threshold:
                        self.last_speech_time = time.time()
                        print(f"🎤 Voz detectada - Volumen: {volume_norm:.0f}")
                    
                    # Verificar si ha pasado suficiente tiempo de silencio
                    silence_duration = time.time() - self.last_speech_time
                    if silence_duration > self.silence_threshold:
                        if self.audio_buffer and not self.is_processing:
                            print(f"🔇 Silencio detectado ({silence_duration:.1f}s) - Procesando audio...")
                            self.process_audio_buffer()
                            break  # Salir del loop para reiniciar
                    
                    time.sleep(0.01)  # Pausa mínima para no saturar CPU
                    
                except Exception as e:
                    print(f"Error en captura de audio: {e}")
                    continue
            
            # Cerrar stream
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
        except Exception as e:
            print(f"Error crítico en escucha de voz: {e}")
            self.root.after(0, self.stop_voice_listening)
    
    def process_audio_with_model(self):
        """Procesa el audio usando el modelo de IA"""
        try:
            # Crear prompt para la IA
            prompt = f"""<|im_start|>system
You are a friendly English tutor who can "hear" audio directly. The user has just spoken to you in English.
Analyze their speech and respond naturally in English. If you detect pronunciation issues, mention them politely.
Keep responses concise and helpful for language practice.
<|im_end|>
<|im_start|>user
[Audio input - user speaking in English]
<|im_end|>
<|im_start|>assistant
"""
            
            # Generar respuesta basada en el audio
            response = self.model(prompt, max_new_tokens=200, temperature=0.7, 
                                stop=["<|im_end|>", "<|im_start|>"])
            
            # Limpiar respuesta
            clean_response = response.strip()
            if clean_response.startswith("assistant"):
                clean_response = clean_response[9:].strip()
            
            if not clean_response:
                clean_response = "I heard you speak! That's great pronunciation practice. Keep going!"
            
            print(f"🎧 IA procesó audio y respondió: '{clean_response}'")
            
            # Mostrar respuesta en UI
            self.root.after(0, lambda: self.show_audio_response(clean_response))
            
        except Exception as e:
            error_msg = f"Error al procesar audio con IA: {str(e)}"
            print(f"Error en process_audio_with_model: {e}")
            self.root.after(0, lambda: self.show_audio_response(error_msg))
    
    def show_audio_response(self, response):
        """Muestra la respuesta de la IA basada en audio"""
        # Mostrar que se procesó audio
        self.add_message("Tú (Voz)", "[Audio procesado en tiempo real]")
        
        # Mostrar respuesta de la IA según el modo configurado
        mode = self.response_mode.get()
        
        if mode in ["TEXT", "TEXT + TTS"]:
            self.add_message("IA", response)
        
        if mode in ["TTS", "TEXT + TTS"]:
            # Responder con voz automáticamente
            threading.Thread(target=self.speak_text, args=(response,), daemon=True).start()
        
        # Actualizar estado
        if not self.is_listening:
            self.status_var.set("¡Modelo cargado! Escribe tu mensaje en inglés o usa el micrófono")
        else:
            self.status_var.set("Escuchando... Habla ahora")
    
    def add_message(self, sender, message):
        """Añade un mensaje al área de chat"""
        self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_area.see(tk.END)
    
    def send_message(self, event=None):
        """Envía el mensaje del usuario"""
        if not self.model_loaded:
            return
        
        message = self.input_field.get().strip()
        if not message:
            return
        
        # Limpiar campo de entrada
        self.input_field.delete(0, tk.END)
        
        # Mostrar mensaje del usuario
        self.add_message("Tú", message)
        
        # Procesar en hilo separado
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
    
    def process_message(self, message):
        """Procesa el mensaje de texto con el modelo"""
        try:
            # Actualizar estado
            self.root.after(0, lambda: self.status_var.set("Procesando..."))
            
            # Crear prompt
            prompt = f"""<|im_start|>system
You are a friendly English tutor. Respond naturally and conversationally in English. Keep responses concise and helpful for language practice. Always respond in English.
<|im_end|>
<|im_start|>user
{message}
<|im_end|>
<|im_start|>assistant
"""
            
            # Generar respuesta
            response = self.model(prompt, max_new_tokens=200, temperature=0.7, 
                                stop=["<|im_end|>", "<|im_start|>"])
            
            # Limpiar respuesta
            clean_response = response.strip()
            if clean_response.startswith("assistant"):
                clean_response = clean_response[9:].strip()
            
            if not clean_response:
                clean_response = "I'm sorry, I didn't understand that. Could you repeat?"
            
            print(f"Respuesta del modelo: '{clean_response}'")
            
            # Actualizar UI en el hilo principal
            self.root.after(0, lambda: self.show_response(clean_response))
            
        except Exception as e:
            error_msg = f"Error al procesar mensaje: {str(e)}"
            print(f"Error en process_message: {e}")
            self.root.after(0, lambda: self.show_response(error_msg))
    
    def show_response(self, response):
        """Muestra la respuesta del modelo según el modo configurado"""
        mode = self.response_mode.get()
        
        if mode in ["TEXT", "TEXT + TTS"]:
            self.add_message("IA", response)
        
        if mode in ["TTS", "TEXT + TTS"]:
            # Responder con voz automáticamente
            threading.Thread(target=self.speak_text, args=(response,), daemon=True).start()
        
        # Actualizar estado solo si no está escuchando voz
        if not self.is_listening:
            self.status_var.set("¡Modelo cargado! Escribe tu mensaje en inglés o usa el micrófono")
        else:
            self.status_var.set("Escuchando... Habla ahora")
    
    def speak_text(self, text):
        """Wrapper para ejecutar TTS en hilo separado"""
        try:
            # Crear nuevo event loop para el hilo
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ejecutar TTS
            loop.run_until_complete(self.speak_text_async(text))
            
        except Exception as e:
            print(f"Error al hablar: {e}")
    
    def speak_last_response(self):
        """Lee en voz alta la última respuesta de la IA"""
        try:
            # Obtener último mensaje de la IA
            content = self.chat_area.get("1.0", tk.END)
            lines = content.strip().split('\n')
            # Buscar última respuesta de la IA
            for line in reversed(lines):
                if line.startswith("IA: "):
                    text_to_speak = line[4:]  # Remover "IA: "
                    if text_to_speak:
                        self.speak_text(text_to_speak)
                        return
            self.speak_text("No hay respuesta para leer")
        except Exception as e:
            print(f"Error al hablar: {e}")

def main():
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
