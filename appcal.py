import streamlit as st

st.title("🧮 Calculator App")

# รับค่าตัวเลข
num1 = st.number_input("Number 1", value=0.0)
num2 = st.number_input("Number 2", value=0.0)

# เลือกเครื่องหมาย
operation = st.selectbox(
    "Choose operation",
    ["+", "-", "*", "/"]
)

# ปุ่มคำนวณ
if st.button("Calculate"):

    if operation == "+":
        result = num1 + num2

    elif operation == "-":
        result = num1 - num2

    elif operation == "*":
        result = num1 * num2

    elif operation == "/":
        if num2 == 0:
            st.error("Cannot divide by zero!")
        else:
            result = num1 / num2

    if operation != "/" or num2 != 0:
        st.success(f"Result = {result}")
