import streamlit as st
from groq import Groq
import re
import io
import math
from streamlit_mic_recorder import mic_recorder
import os
from dotenv import load_dotenv
load_dotenv()

# --- 1. CONFIG & CLIENT ---
st.set_page_config(page_title="Interview Buddy", page_icon="🎤")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_LLM = "llama-3.3-70b-versatile"
MODEL_STT = "whisper-large-v3"

# --- ADVANCED ANALYSIS FUNCTIONS ---
def process_audio(audio_bytes):
    """
    Transcribes audio AND extracts 'hidden' metadata like 
    confidence scores and duration for analysis.
    """
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    
    # 1. Get Verbose JSON for deep analytics
    response = client.audio.transcriptions.create(
        file=audio_file,
        model=MODEL_STT,
        response_format="verbose_json"  # <--- KEY CHANGE
    )
    
    text = response.text
    duration = response.duration
    
    # 2. Calculate Confidence (Logprob -> 0-100% Score)
    # Whisper returns negative log probability. 0 is perfect, -1 is okay, -5 is bad.
    avg_logprob = response.segments[0]['avg_logprob'] if response.segments else -1.0
    confidence_score = math.exp(avg_logprob) * 100 
    
    # 3. Calculate Pace (Words Per Minute)
    word_count = len(text.split())
    wpm = (word_count / duration) * 60 if duration > 0 else 0
    
    return text, duration, confidence_score, wpm

def get_grammar_check(text):
    """Asks LLM to fix grammar and suggest better phrasing."""
    prompt = f"""
    Act as a strict English coach. Analyze this candidate's interview answer:
    "{text}"
    
    Provide output in this exact format:
    1. CORRECTION: [Rewrite the sentence with perfect grammar]
    2. TIP: [One specific tip to sound more professional]
    """
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_LLM
    )
    return resp.choices[0].message.content

# --- UI LAYOUT ---
st.title("👨‍💼 Interview Pro: AI Coach")
st.caption("Analyzes: Content + Clarity + Pace + Grammar")

# Initialize State
if "history" not in st.session_state: st.session_state.history = []
if "q" not in st.session_state: st.session_state.q = "Tell me about yourself and why you want this role."

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.info(f"🗣️ **Current Question:**\n\n{st.session_state.q}")
    audio = mic_recorder(start_prompt="⏺️ Record Answer", stop_prompt="⏹️ Stop & Analyze", key='recorder')

with col2:
    if audio:
        with st.spinner("🎧 Listening & Analyzing metrics..."):
            # Run the complex processing
            text, duration, confidence, wpm = process_audio(audio['bytes'])
            grammar_feedback = get_grammar_check(text)
            
            # --- DASHBOARD METRICS ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Speaking Pace", f"{int(wpm)} WPM", help="Target: 130-150 WPM")
            m2.metric("Clarity Score", f"{int(confidence)}%", help="Based on AI confidence. <80% means you might be mumbling.")
            m3.metric("Duration", f"{duration:.1f}s")
            
            # --- PRONUNCIATION / GRAMMAR SECTION ---
            st.divider()
            st.subheader("📝 Grammar & Phrasing Coach")
            st.warning(f"You said: '{text}'")
            st.markdown(grammar_feedback)
            
            # Save context for follow-up
            st.session_state.history.append({"q": st.session_state.q, "a": text})
            
            # Generate Follow-up
            context_prompt = f"History: {st.session_state.history[-3:]}. Ask a tough follow-up question."
            new_q = client.chat.completions.create(
                 messages=[{"role": "user", "content": context_prompt}],
                 model=MODEL_LLM
            ).choices[0].message.content
            
            if st.button("Next Question ➡️"):
                st.session_state.q = new_q
                st.rerun()