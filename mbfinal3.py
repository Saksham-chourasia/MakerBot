import os
import warnings
import logging
import time
import re
import threading
import subprocess

import pandas as pd
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
import pygame
import requests
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory

try:
    import speedtest
except ImportError:
    speedtest = None


def now():
    return time.perf_counter()


# --------------------- LOG + WARNINGS CLEANUP ---------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --------------------- CONSTANTS ---------------------
WAKE_WORDS = [
    'hey nova', 'hi nova', 'hello nova', 'hoi nova',
    'nova', 'innova', 'inova',
    'hey innova', 'hi innova', 'hello innova',
    'hey inova', 'hi inova', 'hello inova',
    'no va', 'noah', 'hey noah', 'hi noah', 'hello noah',
    'noba', 'hey noba', 'hi noba', 'hello noba',
    'nava', 'hey nava', 'hi nava', 'hello nava',
    'novaa', 'hey novaa', 'hi novaa', 'hello novaa'
]

DOMAIN_KEYWORDS = [
    'maker bhavan', 'maker bhawan', 'maker bhaven', 'inventics', 'invent x', 'maker bhavans',
    'iitgn', 'iit gandhinagar', 'iit gandhi nagar', 'iitg',
    '3d printing', '3d print', '3d printer',
    'workshop', 'workshops',
    'faculty lead', 'faculty leader',
    'shivang sharma',
    'abhi raval', 'abhi rawal', 'abhiii raval', 'abhii raval',
    'pratik mutha', 'prateek mutha', 'prateek muttha', 'pratik muttha', 'pratik', 'prof mutha',
    'prateek mutta', 'professor mutha', 'professor pratik mutha',
    'aniruddh mali', 'anirudh mali',
    'invention factory', 'divij yadav', 'divij yadavs', 'divij',
    'innovation', 'innovations',
    'prototyping', 'prototype', 'prototypings',
    'inventx', 'madhu vadali',
    'tinkerers lab',
    'vishwakarma award',
    'leap program',
    'skill builder', 'skills builder',
    'sprint workshop', 'sprint workshops',
    'summer student fellowship',
    'industry engagement',
    'maker competition', 'maker competitions',
    'electronics prototyping', 'electronics prototyping zone', 'electronic prototyping',
    'pcb milling', 'pcb mill',
    'metal 3d printing',
    'fused deposition modeling',
    'sla printing', 'sla print',
    'laser cutting',
    'vacuum forming',
    'cnc', 'cnc machine',
    'digital fabrication', 'digital fabricate',
    'interactive design lounge',
    'collaborative classroom',
    'project-based learning',
    'active learning',
    'experiential education',
    'reverse engineering',
    'safety training',
    'project officer', 'project offcer',
    'project management', 'project manager',
    'mentorship',
    'incubation', 'incubator',
    'startup support', 'startup supports',
    'student startups',
    'patent filing', 'patent file',
    'innovation challenge',
    'maker culture', 'maker cultures',
    'hands-on learning', 'hands on learning',
    'hemant kanakia',
    'damayanti bhattacharya',
    'maker bhavan foundation',
    'science awareness program', 'science awareness',
    'indian academic makerspaces summit', 'iam summit',
    'center for essential skills',
    'design thinking',
    'entrepreneurship',
    'collaboration', 'collaborations',
    'interdisciplinary learning',
    'mechanical fabrication',
    'equipment booking',
    'technology transfer office', 'tech transfer office',
    'project proposal',
    'industry collaboration',
    'faculty collaboration', 'faculty collaborate',
    'mentor',
    'patron',
]

# --------------------- API & ENV SETUP ---------------------
load_dotenv(".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)

# --------------------- AUDIO + RECOGNITION SETUP ---------------------
pygame.mixer.init()
recognizer = sr.Recognizer()


# --------------------- LIGHTWEIGHT INTERNET SPEED (not shown in table now) ---------------------
def get_internet_speed():
    url = "https://www.google.com/generate_204"
    try:
        t0 = time.time()
        resp = requests.get(url, timeout=2)
        t1 = time.time()
        ping_ms = (t1 - t0) * 1000.0
        size_bytes = len(resp.content) if resp.content is not None else 0
        if t1 - t0 > 0 and size_bytes > 0:
            download_mbps = (size_bytes * 8.0) / (t1 - t0) / 1e6
        else:
            download_mbps = 0.0
        return {
            "ping_ms": ping_ms,
            "download_mbps": download_mbps,
            "upload_mbps": 0.0,
        }
    except Exception:
        return None


# --------------------- ASYNC SPEEDTEST (your original logs) ---------------------
def measure_internet_speed_async(log_func=None):
    def worker():
        if speedtest is None:
            if log_func:
                log_func("🌐 Internet speed test not available. Install with: pip install speedtest-cli")
            return
        try:
            st = speedtest.Speedtest()
            st.get_best_server()
            download_bps = st.download()
            upload_bps = st.upload()
            ping_ms = st.results.ping
            if log_func:
                log_func(
                    f"🌐 Internet Speed (approx) → "
                    f"Ping: {ping_ms:.1f} ms | "
                    f"Download: {download_bps/1e6:.2f} Mbps | "
                    f"Upload: {upload_bps/1e6:.2f} Mbps"
                )
        except Exception as e:
            if log_func:
                log_func(f"🌐 Internet Speed Check Failed: {e}")

    threading.Thread(target=worker, daemon=True).start()


# --------------------- AUDIO INPUT + STT ---------------------
def get_voice_input(filename='output.wav', duration=4, samplerate=44100,
                    log_func=None, return_timings=False):
    if log_func:
        log_func("\n🎙️ Listening... Speak now.")
    else:
        print("\n🎙️ Listening... Speak now.")

    measure_internet_speed_async(log_func)

    audio_wall_start = time.time()
    t_audio_start = now()
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='int16')
    sd.wait()
    t_audio_end = now()
    audio_wall_end = time.time()
    audio_time = t_audio_end - t_audio_start

    sf.write(filename, audio, samplerate)

    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)

        stt_wall_start = time.time()
        t_stt_start = now()
        try:
            text = recognizer.recognize_google(audio_data, language="en-IN")
            t_stt_end = now()
            stt_wall_end = time.time()
            stt_time = t_stt_end - t_stt_start

            if log_func:
                log_func(f"⏱️ Audio Record Time: {audio_time:.2f} sec")
                log_func(f"⏱️ STT Time: {stt_time:.2f} sec")
                log_func(f"🗣️ You said: {text}")
            if return_timings:
                return text, {
                    "audio_wall_start": audio_wall_start,
                    "audio_wall_end": audio_wall_end,
                    "audio_time": audio_time,
                    "stt_wall_start": stt_wall_start,
                    "stt_wall_end": stt_wall_end,
                    "stt_time": stt_time,
                }
            return text

        except Exception:
            t_stt_end = now()
            stt_wall_end = time.time()
            stt_time = t_stt_end - t_stt_start
            if log_func:
                log_func(f"⏱️ Audio Record Time: {audio_time:.2f} sec")
                log_func(f"⏱️ STT Time (error): {stt_time:.2f} sec")
                log_func("❌ Could not understand audio.")
            if return_timings:
                return None, {
                    "audio_wall_start": audio_wall_start,
                    "audio_wall_end": audio_wall_end,
                    "audio_time": audio_time,
                    "stt_wall_start": stt_wall_start,
                    "stt_wall_end": stt_wall_end,
                    "stt_time": stt_time,
                }
            return None


def clean_text_for_speech(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"[#*_~><|\\/\[\]{}]", "", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def process_jsonl_to_documents(file_path):
    documents = []
    data = pd.read_json(file_path, lines=True)
    for _, row in data.iterrows():
        system_prompt = row['messages'][0]['content']
        question = row['messages'][1]['content']
        answer = row['messages'][2]['content']
        documents.append(
            Document(
                page_content=f"Question: {question}\nAnswer: {answer}",
                metadata={
                    "source": "maker_bhavan_dataset",
                    "topic": system_prompt
                }
            )
        )
    return documents


# --------------------- MAIN CHATBOT CLASS ---------------------
class ImprovedChatbot:
    def __init__(self, log_callback=None):
        t_init_start = now()

        self.log_callback = log_callback
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="question"
        )

        self._log("⚙️ Initializing embeddings + vectorstore (one-time)...")
        t_vec_start = now()
        self.vectorstore = self.initialize_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        t_vec_end = now()
        self._log(f"⏱️ Vectorstore Init Time: {t_vec_end - t_vec_start:.2f} sec")

        self.last_llm_metrics = None
        self.last_tts_metrics = None

        t_init_end = now()
        self._log(f"✅ Chatbot Ready (Total init time: {t_init_end - t_init_start:.2f} sec)")

    def _log(self, msg: str):
        print(msg) if self.log_callback is None else self.log_callback(msg)

    def initialize_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db_path = "./chroma_db"
        if os.path.exists(db_path):
            return Chroma(persist_directory=db_path, embedding_function=embeddings)
        else:
            docs = process_jsonl_to_documents("dataset.jsonl")
            return Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=db_path
            )

    def _format_chat_history(self, chat_history):
        return "\n".join(
            f"User: {msg.content}" if msg.type == "human"
            else f"Assistant: {msg.content}"
            for msg in chat_history[-4:]
        )

    def is_domain_query(self, query: str) -> bool:
        q = query.lower()
        return any(keyword in q for keyword in DOMAIN_KEYWORDS)

    # --------------------- GEMINI CALL (2.5 FLASH) ---------------------
    def _ask_gemini(self, prompt: str) -> str:
        llm_wall_start = time.time()
        self.last_llm_metrics = None

        try:
            if not GEMINI_API_KEY:
                self._log("❌ No GEMINI_API_KEY found in .env")
                return (
                    "I don't have access to the online AI service right now, "
                    "but I can still answer Maker Bhavan questions from my local knowledge."
                )

            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            }

            resp = requests.post(
                GEMINI_API_URL,
                headers=headers,
                json=payload,
                timeout=20
            )
            llm_wall_end = time.time()

            data = resp.json() if resp.status_code == 200 else {}

            candidates = data.get("candidates", [])
            for cand in candidates:
                parts = cand.get("content", {}).get("parts", [])
                texts = [p.get("text", "") for p in parts if "text" in p]
                if texts:
                    answer = " ".join(texts).strip()
                    self.last_llm_metrics = {
                        "llm_wall_start": llm_wall_start,
                        "llm_wall_end": llm_wall_end,
                    }
                    return answer or "I'm sorry, I couldn't generate a response."

            self.last_llm_metrics = {
                "llm_wall_start": llm_wall_start,
                "llm_wall_end": llm_wall_end,
            }
            return "I'm sorry, I couldn't generate a response."

        except Exception:
            llm_wall_end = time.time()
            self.last_llm_metrics = {
                "llm_wall_start": llm_wall_start,
                "llm_wall_end": llm_wall_end,
            }
            return "I'm sorry, I ran into an issue while generating an online response."

    # --------------------- RAG ANSWER ---------------------
    def answer_domain(self, question: str) -> str:
        self.last_llm_metrics = None
        try:
            t_rag_start = now()
            docs = self.retriever.get_relevant_documents(question)
            t_rag_end = now()
            self._log(f"⏱️ RAG Retrieval Time (offline): {t_rag_end - t_rag_start:.2f} sec")

            if not docs:
                return "I couldn't find information about that in the Maker Bhavan dataset."

            best_doc = docs[0]
            text = best_doc.page_content or ""
            m = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
            if m:
                ans = m.group(1).strip()
                if ans:
                    return ans
            return (
                "I found related information in the Maker Bhavan dataset, "
                "but couldn't extract a precise answer."
            )
        except Exception as e:
            self._log(f"❌ Domain answer error: {e}")
            return "Something went wrong while looking up the Maker Bhavan information."

    def answer_general(self, question: str) -> str:
        chat_history = self._format_chat_history(
            self.memory.load_memory_variables({}).get("chat_history", [])
        )
        prompt = (
            "You are a helpful general-purpose assistant. "
            "You speak in clear Indian English. "
            "Answer briefly and to the point.\n\n"
            f"{chat_history}\n"
            f"User: {question}\n"
            "Assistant:"
        )
        return self._ask_gemini(prompt)

    # --------------------- TTS ---------------------
    def speak(self, text: str):
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            return
        self._log("🔊 Speaking...")

        tts_wall_start = time.time()
        self.last_tts_metrics = None

        try:
            gtts_wall_start = time.time()
            tts = gTTS(cleaned_text, slow=False, lang='en', tld='co.in')
            tts.save("response.mp3")
            gtts_wall_end = time.time()
            gtts_time = gtts_wall_end - gtts_wall_start

            playback_wall_start = time.time()
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            pygame.mixer.music.unload()
            os.remove("response.mp3")
            playback_wall_end = time.time()
            playback_time = playback_wall_end - playback_wall_start

            tts_wall_end = time.time()
            tts_total = tts_wall_end - tts_wall_start

            self.last_tts_metrics = {
                "tts_wall_start": tts_wall_start,
                "tts_wall_end": tts_wall_end,
                "tts_total": tts_total,
                "gtts_wall_start": gtts_wall_start,
                "gtts_wall_end": gtts_wall_end,
                "gtts_time": gtts_time,
                "playback_wall_start": playback_wall_start,
                "playback_wall_end": playback_wall_end,
                "playback_time": playback_time,
            }

        except Exception as e:
            tts_wall_end = time.time()
            self._log(f"❌ gTTS playback error: {e}")
            self.last_tts_metrics = {
                "tts_wall_start": tts_wall_start,
                "tts_wall_end": tts_wall_end,
                "tts_total": tts_wall_end - tts_wall_start,
                "gtts_wall_start": None,
                "gtts_wall_end": None,
                "gtts_time": None,
                "playback_wall_start": None,
                "playback_wall_end": None,
                "playback_time": None,
            }

    def ask_llama_cpp(self, prompt):
        llama_path = "/home/pi/Desktop/makerbot/MakerBot2.0/llama.cpp/build/bin/llama-cli"
        model_path = os.path.expanduser("~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        try:
            result = subprocess.run(
                [llama_path, "-m", model_path, "-p", prompt],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error running llama.cpp: {e}"

    # --------------------- TABLE HELPERS ---------------------
    def _format_clock(self, ts):
        if ts is None:
            return "N/A"
        return time.strftime("%H:%M:%S", time.localtime(ts))

    def _format_duration(self, seconds):
        if seconds is None:
            return "N/A"
        mins = int(seconds // 60)
        secs = seconds - mins * 60
        if mins > 0:
            return f"{mins} min {secs:.2f} sec"
        else:
            return f"{secs:.2f} sec"

    def _print_timing_table(self, stt_timings):
        audio_start = stt_timings["audio_wall_start"]
        audio_end = stt_timings["audio_wall_end"]
        audio_time = stt_timings["audio_time"]

        stt_start = stt_timings["stt_wall_start"]
        stt_end = stt_timings["stt_wall_end"]
        stt_time = stt_timings["stt_time"]

        llm = self.last_llm_metrics
        if llm is not None:
            llm_start = llm["llm_wall_start"]
            llm_end = llm["llm_wall_end"]
            llm_time = llm_end - llm_start
        else:
            llm_start = llm_end = llm_time = None

        tts = self.last_tts_metrics
        if tts is not None:
            tts_start = tts["tts_wall_start"]
            tts_end = tts["tts_wall_end"]
            tts_total = tts["tts_total"]
            gtts_start = tts["gtts_wall_start"]
            gtts_end = tts["gtts_wall_end"]
            gtts_time = tts["gtts_time"]
            play_start = tts["playback_wall_start"]
            play_end = tts["playback_wall_end"]
            play_time = tts["playback_time"]
            if play_start is not None:
                gap_start = stt_end
                gap_end = play_start
                gap_time = gap_end - gap_start
            else:
                gap_start = gap_end = gap_time = None
        else:
            tts_start = tts_end = tts_total = None
            gtts_start = gtts_end = gtts_time = None
            play_start = play_end = play_time = None
            gap_start = gap_end = gap_time = None

        self._log("\n================ TIMING SUMMARY (per turn) ================")
        header = f"{'Phase':<32} {'Start Time':>12} {'End Time':>12} {'Duration':>18}"
        self._log(header)
        self._log("-" * len(header))

        self._log(
            f"{'Mic Recording':<32} "
            f"{self._format_clock(audio_start):>12} "
            f"{self._format_clock(audio_end):>12} "
            f"{self._format_duration(audio_time):>18}"
        )
        self._log(
            f"{'Speech-to-Text (Google STT)':<32} "
            f"{self._format_clock(stt_start):>12} "
            f"{self._format_clock(stt_end):>12} "
            f"{self._format_duration(stt_time):>18}"
        )
        self._log(
            f"{'Gap: STT end → Speech start':<32} "
            f"{self._format_clock(gap_start):>12} "
            f"{self._format_clock(gap_end):>12} "
            f"{self._format_duration(gap_time):>18}"
        )
        self._log(
            f"{'LLM Request (Gemini API)':<32} "
            f"{self._format_clock(llm_start):>12} "
            f"{self._format_clock(llm_end):>12} "
            f"{self._format_duration(llm_time):>18}"
        )
        self._log(
            f"{'TTS Synthesis (gTTS)':<32} "
            f"{self._format_clock(gtts_start):>12} "
            f"{self._format_clock(gtts_end):>12} "
            f"{self._format_duration(gtts_time):>18}"
        )
        self._log(
            f"{'TTS Playback (speaker)':<32} "
            f"{self._format_clock(play_start):>12} "
            f"{self._format_clock(play_end):>12} "
            f"{self._format_duration(play_time):>18}"
        )
        self._log(
            f"{'TTS Overall (synth + play)':<32} "
            f"{self._format_clock(tts_start):>12} "
            f"{self._format_clock(tts_end):>12} "
            f"{self._format_duration(tts_total):>18}"
        )

        self._log("===========================================================\n")

    # --------------------- CHAT LOOP ---------------------
    def chat_interface(self, require_wake_word=True):
        self._log("🤖 Enhanced Hybrid Chatbot Initialized (Say 'exit' to quit)")
        if require_wake_word:
            while True:
                self._log("\n🔍 Waiting for wake word...")
                query, _ = get_voice_input(log_func=self._log, return_timings=True)
                if not query:
                    continue
                if query.lower() == 'exit':
                    self._log("👋 Goodbye!")
                    break
                if any(w in query.lower() for w in WAKE_WORDS):
                    self.speak("Hello! How can I assist you?")
                    self.active_conversation()
                    break
                else:
                    self._log("👂 Waiting for the correct wake word...")
        else:
            self.active_conversation()

    def active_conversation(self):
        session_active = True
        session_start_time = time.time()

        while session_active:
            self._log("\n🎙️ Listening for your query...")
            query, stt_timings = get_voice_input(
                log_func=self._log,
                return_timings=True
            )

            if not query:
                if time.time() - session_start_time > 30:
                    self.speak("Do you want to continue the conversation?")
                    confirm, _ = get_voice_input(log_func=self._log, return_timings=True)
                    if confirm and any(word in confirm.lower() for word in ['no', 'exit', 'bye']):
                        self.speak("Have a good day!")
                        session_active = False
                        break
                    else:
                        session_start_time = time.time()
                continue

            self._log(
                f"⏱️ Phase STT → Audio Record: {stt_timings['audio_time']:.2f} sec, "
                f"Speech-to-Text: {stt_timings['stt_time']:.2f} sec"
            )

            if query.lower() in ['exit', 'bye', 'goodbye', 'no']:
                self.speak("Have a good day!")
                session_active = False
                break

            # llama.cpp integration: if query starts with 'llama:'
            if query.lower().startswith("llama:"):
                llama_prompt = query[6:].strip()
                response = self.ask_llama_cpp(llama_prompt)
                self._log(f"llama.cpp: {response}")
                self.speak(response)
                self._print_timing_table(stt_timings)
                session_start_time = time.time()
                continue

            try:
                if self.is_domain_query(query):
                    self._log("\nBot (RAG / Maker Bhavan – offline):")
                    response = self.answer_domain(query)
                else:
                    self._log("\nBot (General – online Gemini 2.5 Flash):")
                    response = self.answer_general(query)

                self.memory.save_context({"question": query}, {"answer": response})
                self._log(response)

                self.speak(response)

                # Show per-turn timing table
                self._print_timing_table(stt_timings)

                session_start_time = time.time()

            except Exception as e:
                self._log(f"❌ Error: {e}")


# --------------------- ENTRY POINT ---------------------
if __name__ == "__main__":
    chatbot = ImprovedChatbot()
    chatbot.chat_interface()
