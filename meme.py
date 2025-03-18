#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import torch
import re
import sympy
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the trained model and tokenizer
MODEL_PATH = './fine_tuned_math_meme_model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        body {
            background-color: #e0f7fa;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            margin: auto;
            max-width: 700px;
        }
        .stButton>button {
            background-color: #4fc3f7;
            color: white;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            transition: 0.3s;
            border: none;
        }
        .stButton>button:hover {
            background-color: #039be5;
        }
        h1 {
            color: #0277bd !important;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
        }
        h3 {
            color: #01579b;
            text-align: center;
            font-size: 22px;
        }
        p {
            color: #0288d1;
            font-weight: bold;
            text-align: center;
            font-size: 18px;
        }
        .stTextInput>div>div>input {
            border: 2px solid #0288d1;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app UI
st.markdown("<h1 style='color: #0277bd;'>Math Meme Repair</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #0288d1; font-size: 18px;'>Check if your math meme is correct and get explanations!</p>", unsafe_allow_html=True)

meme_input = st.text_input("Enter a math meme (e.g., '8 + 4 ÷ 2 × 3 = 18?'): ")

def evaluate_expression(expression: str) -> float:
    """Evaluates the correct result using SymPy."""
    try:
        expr = expression.replace('^', '**').replace('÷', '/').replace('×', '*')
        expr = re.sub(r'(?<=\d)\(', '*(', expr)
        return float(sympy.sympify(expr))
    except Exception:
        return None

def analyze_meme(meme: str):
    """Analyze and correct a math meme."""
    inputs = tokenizer(meme, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    try:
        expression, claimed_result = meme.split('=')
        claimed_result = float(claimed_result.strip().replace('?', ''))
        correct_result = evaluate_expression(expression.strip())
        
        if correct_result is None:
            return "Invalid Expression", "Cannot evaluate. Please check your input."
        
        if prediction == 0:
            return (f"❌ Wrong: {meme} → ✅ Correct: {expression} = {correct_result}",
                    "Follow PEMDAS: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction")
        return (f"✅ {meme}", "This math is accurate!")
    except:
        return "Error parsing meme", "Cannot process the input."

if st.button("Check Meme"):
    if meme_input:
        result, explanation = analyze_meme(meme_input)
        st.markdown(f"<h3>{result}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>{explanation}</p>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a math meme to analyze.")
