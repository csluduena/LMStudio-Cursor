import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from ctransformers import AutoModelForCausalLM
import pyttsx3
import queue
import speech_recognition as sr
import time
import pyaudio
import numpy as np
from collections import deque

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat con IA Local - Práctica de Inglés")
        self.root.geometry("1200x700")  # Ventana más ancha para el panel
        
        # Cola para comunicación entre hilos
        self.message_queue = queue.Queue()
        
        # Inicializar motor de voz
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # Inicializar reconocimiento de voz
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Variables para reconocimiento de voz en tiempo real
        self.is_listening = False
        self.voice_thread = None
        self.audio_buffer = deque(maxlen=50)  # Buffer de audio
        self.last_audio_time = 0
        self.silence_threshold = 2.0  # segundos de silencio para detectar fin de frase
        self.min_phrase_duration = 0.5  # duración mínima de una frase
        self.is_processing_audio = False
        
        # Configuración de respuesta
        self.response_mode = tk.StringVar(value="TEXT + TTS")  # TEXT, TTS, TEXT + TTS
        
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
        main_container.columnconfigure(0, weight=3)  # Chat area
        main_container.columnconfigure(1, weight=1)  # Panel de configuración
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
                       value="TEXT").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(response_frame, text="Solo Voz (TTS)", variable=self.response_mode, 
                       value="TTS").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(response_frame, text="Texto + Voz", variable=self.response_mode, 
                       value="TEXT + TTS").grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # === CONFIGURACIÓN DE VOZ ===
        voice_config_frame = ttk.LabelFrame(config_frame, text="🎤 Configuración de Voz", padding="5")
        voice_config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Umbral de silencio
        ttk.Label(voice_config_frame, text="Pausa para fin de frase:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        silence_frame = ttk.Frame(voice_config_frame)
        silence_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.silence_slider = ttk.Scale(silence_frame, from_=1.0, to=10.0, 
                                       orient=tk.HORIZONTAL, length=150,
                                       command=self.update_silence_threshold)
        self.silence_slider.set(self.silence_threshold)
        self.silence_slider.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.silence_label = ttk.Label(silence_frame, text=f"{self.silence_threshold:.1f}s")
        self.silence_label.grid(row=0, column=1, padx=(5, 0))
        
        # Velocidad de habla
        ttk.Label(voice_config_frame, text="Velocidad de habla:").grid(row=2, column=0, sticky=tk.W, pady=(10, 2))
        
        speed_frame = ttk.Frame(voice_config_frame)
        speed_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.speed_slider = ttk.Scale(speed_frame, from_=50, to=300, 
                                     orient=tk.HORIZONTAL, length=150,
                                     command=self.update_speech_speed)
        self.speed_slider.set(150)
        self.speed_slider.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.speed_label = ttk.Label(speed_frame, text="150")
        self.speed_label.grid(row=0, column=1, padx=(5, 0))
        
        # Volumen
        ttk.Label(voice_config_frame, text="Volumen:").grid(row=4, column=0, sticky=tk.W, pady=(10, 2))
        
        volume_frame = ttk.Frame(voice_config_frame)
        volume_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.volume_slider = ttk.Scale(volume_frame, from_=0.0, to=1.0, 
                                      orient=tk.HORIZONTAL, length=150,
                                      command=self.update_volume)
        self.volume_slider.set(0.9)
        self.volume_slider.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.volume_label = ttk.Label(volume_frame, text="0.9")
        self.volume_label.grid(row=0, column=1, padx=(5, 0))
        
        # === INFORMACIÓN DEL SISTEMA ===
        info_frame = ttk.LabelFrame(config_frame, text="ℹ️ Información", padding="5")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(info_frame, text="Modelo: OpenHermes-2.5-Mistral-7B", 
                 font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(info_frame, text="Idioma: Inglés", 
                 font=("Arial", 9)).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(info_frame, text="Reconocimiento: Google Speech", 
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
        volume_frame.columnconfigure(0, weight=1)
        silence_frame.columnconfigure(0, weight=1)
    
    def update_silence_threshold(self, value):
        """Actualiza el umbral de silencio"""
        self.silence_threshold = float(value)
        self.silence_label.config(text=f"{self.silence_threshold:.1f}s")
    
    def update_speech_speed(self, value):
        """Actualiza la velocidad de habla"""
        speed = int(float(value))
        self.engine.setProperty('rate', speed)
        self.speed_label.config(text=str(speed))
    
    def update_volume(self, value):
        """Actualiza el volumen"""
        volume = float(value)
        self.engine.setProperty('volume', volume)
        self.volume_label.config(text=f"{volume:.1f}")
    
    def reset_config(self):
        """Reinicia la configuración a valores por defecto"""
        self.silence_slider.set(2.0)
        self.speed_slider.set(150)
        self.volume_slider.set(0.9)
        self.response_mode.set("TEXT + TTS")
        
        self.silence_threshold = 2.0
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        self.silence_label.config(text="2.0s")
        self.speed_label.config(text="150")
        self.volume_label.config(text="0.9")
    
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
        self.last_audio_time = time.time()
        
        # Iniciar hilo de escucha
        self.voice_thread = threading.Thread(target=self.continuous_voice_listening, daemon=True)
        self.voice_thread.start()
        
        self.add_message("Sistema", "🎤 Micrófono activado. Habla en inglés para practicar.")
    
    def stop_voice_listening(self):
        """Detiene la escucha de voz"""
        self.is_listening = False
        self.mic_button.config(text="🎤 Iniciar Escucha")
        self.voice_status_var.set("Micrófono desactivado")
        
        # Procesar cualquier audio restante en el buffer
        if self.audio_buffer and not self.is_processing_audio:
            self.process_audio_buffer()
        
        self.add_message("Sistema", "🎤 Micrófono desactivado.")
    
    def continuous_voice_listening(self):
        """Escucha continuamente la voz del usuario con mejor manejo de errores"""
        try:
            with self.microphone as source:
                # Ajustar para ruido ambiental
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                while self.is_listening:
                    try:
                        # Escuchar con timeout muy corto para no bloquear
                        audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=15)
                        
                        # Agregar audio al buffer con timestamp
                        self.audio_buffer.append((audio, time.time()))
                        self.last_audio_time = time.time()
                        
                        # Verificar si hay suficiente silencio para procesar
                        if time.time() - self.last_audio_time > self.silence_threshold:
                            if self.audio_buffer and not self.is_processing_audio:
                                self.process_audio_buffer()
                        
                    except sr.WaitTimeoutError:
                        # No se detectó voz, verificar si hay audio en buffer para procesar
                        if (self.audio_buffer and 
                            time.time() - self.last_audio_time > self.silence_threshold and
                            not self.is_processing_audio):
                            self.process_audio_buffer()
                        continue
                        
                    except sr.UnknownValueError:
                        # Audio no reconocible, continuar escuchando
                        continue
                        
                    except Exception as e:
                        print(f"Error en escucha de voz: {e}")
                        # Continuar escuchando en lugar de crashear
                        continue
                        
        except Exception as e:
            print(f"Error crítico en escucha de voz: {e}")
            self.root.after(0, self.stop_voice_listening)
    
    def process_audio_buffer(self):
        """Procesa el buffer de audio acumulado"""
        if not self.audio_buffer or self.is_processing_audio:
            return
        
        self.is_processing_audio = True
        
        try:
            # Combinar todos los audios del buffer
            combined_audio = self.audio_buffer[0][0]  # Tomar el primer audio como base
            
            # Intentar transcribir
            text = self.recognizer.recognize_google(combined_audio, language='en-US')
            
            if text and text.strip():
                # Limpiar buffer
                self.audio_buffer.clear()
                
                # Procesar en hilo principal
                self.root.after(0, lambda t=text: self.process_voice_input(t))
                
        except sr.UnknownValueError:
            # Audio no reconocible, limpiar buffer
            self.audio_buffer.clear()
        except Exception as e:
            print(f"Error al procesar audio: {e}")
            self.audio_buffer.clear()
        finally:
            self.is_processing_audio = False
    
    def process_voice_input(self, text):
        """Procesa el input de voz del usuario"""
        # Mostrar mensaje del usuario
        self.add_message("Tú (Voz)", text)
        
        # Procesar con el modelo
        threading.Thread(target=self.process_message, args=(text,), daemon=True).start()
    
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
        """Procesa el mensaje con el modelo"""
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
            
            # Actualizar UI en el hilo principal
            self.root.after(0, lambda: self.show_response(clean_response))
            
        except Exception as e:
            error_msg = f"Error al procesar mensaje: {str(e)}"
            self.root.after(0, lambda: self.show_response(error_msg))
    
    def show_response(self, response):
        """Muestra la respuesta del modelo según el modo configurado"""
        mode = self.response_mode.get()
        
        if mode in ["TEXT", "TEXT + TTS"]:
            self.add_message("IA", response)
        
        if mode in ["TTS", "TEXT + TTS"]:
            # Si está escuchando voz, responder automáticamente
            if self.is_listening:
                threading.Thread(target=self.speak_text, args=(response,), daemon=True).start()
        
        self.status_var.set("¡Modelo cargado! Escribe tu mensaje en inglés o usa el micrófono")
    
    def speak_text(self, text):
        """Lee en voz alta el texto"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
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
                        break
            else:
                self.speak_text("No hay respuesta para leer")
                
        except Exception as e:
            print(f"Error al hablar: {e}")

def main():
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
