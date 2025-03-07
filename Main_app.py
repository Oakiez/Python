import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go

# ต้องเรียกใช้ set_page_config() เป็นคำสั่งแรกใน Streamlit
st.set_page_config(
    page_title="ระบบทำนายความเสี่ยงการเกิดโรคเบาหวาน",
    page_icon="?",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ฟังก์ชัน load model
@st.cache_resource
def load_models():
    try:
        models = {}
        scalers = {}
        # สำหรับโมเดลแต่ละแบบ (full, no_bp, no_glucose, minimal)
        for model_type in ['full', 'no_bp', 'no_glucose', 'minimal']:
            with open(f'models/model_{model_type}.pkl', 'rb') as file:
                models[model_type] = pickle.load(file)
            with open(f'models/scaler_{model_type}.pkl', 'rb') as file:
                scalers[model_type] = pickle.load(file)
        return models, scalers
    except FileNotFoundError:
        st.error("ไม่พบไฟล์โมเดล โปรดรันไฟล์ train_model.py ก่อน")
        st.stop()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        st.stop()

# โหลดโมเดลและ scaler
models, scalers = load_models()

# ส่วนหัวของแอพ
st.title("ระบบทำนายความเสี่ยงเบาหวาน")
st.markdown("---")

# แสดงข้อมูลเกี่ยวกับแอพ
with st.expander("ℹ️ เกี่ยวกับระบบนี้", expanded=False):
    st.markdown("""
    **ระบบทำนายความเสี่ยงการเกิดโรคเบาหวาน** นี้ใช้ข้อมูลทางการแพทย์เพื่อวิเคราะห์ความเสี่ยง
    โดยใช้ชุดข้อมูล Pima Indians Diabetes Dataset และอัลกอริทึม Random Forest 
    โดยเป็นส่วนหนึ่งของรายวิชา Computational Science CP351101
    จัดทำโดย นายวงศธร ธน.ยอด และ นางสาวภัทราพร ศรีชนะ

    **คำเตือน**: ระบบนี้เป็นแค่เครื่องมือช่วยคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้ 
    หากมีข้อสงสัยกรุณาปรึกษาแพทย์
    """)

# สร้างแท็บ
tab1, tab2 = st.tabs(["🗃️ กรอกข้อมูลสุขภาพ", "⚖️ คำนวณค่า BMI"])

# แท็บสำหรับคำนวณ BMI
with tab2:
    st.header("คำนวณดัชนีมวลกาย (BMI)")

    col1, col2 = st.columns(2)

    with col1:
        weight = st.number_input("น้ำหนัก (กิโลกรัม)", min_value=0.0, max_value=300.0, value=65.0, step=0.1)

    with col2:
        height_cm = st.number_input("ส่วนสูง (เซนติเมตร)", min_value=0.0, max_value=250.0, value=165.0, step=0.1)

    if st.button("คำนวณ BMI", use_container_width=True):
        if height_cm > 0:
            height_m = height_cm / 100
            bmi_value = weight / (height_m * height_m)
            st.success(f"ค่า BMI ของคุณคือ: {bmi_value:.2f}")

            # แสดงระดับความอ้วน
            if bmi_value < 18.5:
                st.info("น้ำหนักต่ำกว่าเกณฑ์")
            elif bmi_value < 23:
                st.info("น้ำหนักปกติ")
            elif bmi_value < 25:
                st.warning("น้ำหนักเกิน")
            elif bmi_value < 30:
                st.warning("อ้วน")
            else:
                st.error("อ้วนมาก")

            # สร้างปุ่มเพื่อนำค่า BMI ไปใช้ในแท็บหลัก
            if st.button("ใช้ค่า BMI นี้", use_container_width=True):
                st.session_state.bmi = round(bmi_value, 2)
                st.info(f"ค่า BMI {bmi_value:.2f} ถูกบันทึกแล้ว สามารถกลับไปที่แท็บ 'กรอกข้อมูลสุขภาพ' เพื่อทำการทำนาย")
        else:
            st.error("กรุณากรอกส่วนสูงที่มากกว่า 0")

# แท็บหลักสำหรับกรอกข้อมูล
with tab1:
    st.header("กรอกข้อมูลสุขภาพของคุณ")

    # คำถามประเมินความเสี่ยง
    col1, col2 = st.columns(2)

    with col1:
        # เปลี่ยนจำนวนการตั้งครรภ์เป็นคำถามว่าเคยมีบุตรหรือไม่
        has_children = st.radio("เคยมีบุตรมาก่อนหรือไม่", ["ไม่เคย", "เคย"])
        pregnancies = 0 if has_children == "ไม่เคย" else st.number_input("จำนวนบุตร", min_value=1, max_value=20, value=1)

        # ทางเลือกสำหรับระดับน้ำตาล
        glucose_option = st.radio("คุณทราบระดับน้ำตาลในเลือด (มก./ดล.) หรือไม่", ["ไม่ทราบ", "ทราบ"])
        if glucose_option == "ทราบ":
            glucose = st.number_input("ระดับน้ำตาลในเลือด (มก./ดล.)", min_value=0, max_value=300, value=100)
        else:
            glucose = None

        # ค่า BMI จากแท็บคำนวณหรือป้อนเอง
        if 'bmi' in st.session_state:
            bmi = st.number_input("ดัชนีมวลกาย (BMI)", min_value=0.0, max_value=60.0, value=st.session_state.bmi,
                                  step=0.1)
        else:
            bmi = st.number_input("ดัชนีมวลกาย (BMI)", min_value=0.0, max_value=60.0, step=0.1)

    st.info("ไม่ทราบค่า BMI? ไปที่แท็บ 'คำนวณค่า BMI' เพื่อคำนวณ")

    with col2:
        # ทางเลือกสำหรับความดัน
        bp_option = st.radio("คุณทราบความดันโลหิต (มม.ปรอท) หรือไม่", ["ไม่ทราบ", "ทราบ"])
        if bp_option == "ทราบ":
            blood_pressure = st.number_input("ความดันโลหิต (มม.ปรอท)", min_value=0, max_value=200, value=70)
        else:
            blood_pressure = None

        age = st.number_input("อายุ (ปี)", min_value=0, max_value=120)

        # ทำประวัติเบาหวานในครอบครัวให้เข้าใจง่ายขึ้น
        diabetes_family_history = st.selectbox("ประวัติเบาหวานในครอบครัว",
                                               ["ไม่มีประวัติเบาหวานในครอบครัว",
                                                "มีญาติห่างๆ เป็นเบาหวาน",
                                                "มีปู่/ย่า/ตา/ยายเป็นเบาหวาน",
                                                "มีพ่อหรือแม่เป็นเบาหวาน",
                                                "มีพ่อและแม่เป็นเบาหวาน"])

        # แปลงคำตอบเป็นค่า diabetes_pedigree
        if diabetes_family_history == "ไม่มีประวัติเบาหวานในครอบครัว":
            diabetes_pedigree = 0.1
        elif diabetes_family_history == "มีญาติห่างๆ เป็นเบาหวาน":
            diabetes_pedigree = 0.3
        elif diabetes_family_history == "มีปู่/ย่า/ตา/ยายเป็นเบาหวาน":
            diabetes_pedigree = 0.6
        elif diabetes_family_history == "มีพ่อหรือแม่เป็นเบาหวาน":
            diabetes_pedigree = 1.0
        else:
            diabetes_pedigree = 1.5

    # ค่าอื่นๆ ที่ใช้แทนข้อมูลที่ขาด
    insulin = 80  # ค่าเฉลี่ยทั่วไป
    skin_thickness = 20  # ค่าเฉลี่ยทั่วไป (ตัดออกจากฟอร์ม)

    # ปุ่มทำนาย
    predict_button = st.button("ทำนายความเสี่ยง", type="primary", use_container_width=True)

    # เมื่อกดปุ่มทำนาย
    if predict_button:
        # เลือกโมเดลโดยพิจารณาจากข้อมูลที่ขาด
        if glucose is None and blood_pressure is None:
            model_type = 'minimal'
            input_data = np.array(
                [pregnancies, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
        elif glucose is None:
            model_type = 'no_glucose'
            input_data = np.array(
                [pregnancies, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
        elif blood_pressure is None:
            model_type = 'no_bp'
            input_data = np.array(
                [pregnancies, glucose, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
        else:
            model_type = 'full'
            input_data = np.array(
                [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1,
                                                                                                                      -1)

        # ปรับข้อมูล
        input_scaled = scalers[model_type].transform(input_data)

        # ทำนาย
        prediction = models[model_type].predict(input_scaled)
        prediction_proba = models[model_type].predict_proba(input_scaled)

        # แสดงผลลัพธ์
        st.markdown("---")
        st.header("ผลการทำนาย")

        # สร้างกราฟแสดงความน่าจะเป็น
        risk_percent = round(prediction_proba[0][1] * 100, 2)

        if prediction[0] == 1:
            st.error(f"⚠️ พบความเสี่ยงเบาหวาน ({risk_percent}%)")
            st.markdown("""
            **คำแนะนำ**: คุณควรปรึกษาแพทย์เพื่อการตรวจวินิจฉัยที่ละเอียดขึ้น และควรปรับเปลี่ยนพฤติกรรมการใช้ชีวิต
            - ควบคุมอาหาร ลดการบริโภคน้ำตาลและแป้ง
            - ออกกำลังกายอย่างสม่ำเสมอ อย่างน้อย 30 นาทีต่อวัน
            - ควบคุมน้ำหนักให้อยู่ในเกณฑ์ปกติ
            - ตรวจระดับน้ำตาลในเลือดเป็นประจำ
            """)
        else:
            st.success(f"✅ ความเสี่ยงเบาหวานต่ำ ({100 - risk_percent}%)")
            st.markdown("""
            **คำแนะนำ**: แม้ความเสี่ยงต่ำ แต่ควรดูแลสุขภาพอย่างต่อเนื่อง
            - รับประทานอาหารที่มีประโยชน์ หลีกเลี่ยงอาหารหวานและมันมากเกินไป
            - ออกกำลังกายอย่างสม่ำเสมอ 
            - นอนหลับพักผ่อนให้เพียงพอ
            - ตรวจสุขภาพประจำปี
            """)

        # แสดงกราฟวงกลมแสดงความเสี่ยง
        col1, col2 = st.columns([1, 1])

        with col1:
            # สร้างกราฟวงกลมด้วย Plotly
            if prediction[0] == 1:
                colors = ['#a8f0ab', '#ff9999']
                labels = ['ปกติ', 'เสี่ยง']
                values = [100 - risk_percent, risk_percent]
            else:
                colors = ['#a8f0ab', '#ff9999']
                labels = ['ปกติ', 'เสี่ยง']
                values = [100 - risk_percent, risk_percent]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textfont=dict(size=14),
                pull=[0.05, 0]
            )])

            fig.update_layout(
                width=400,
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig)

        with col2:
            # แสดงข้อมูลเพิ่มเติม
            st.subheader("ปัจจัยเสี่ยงของคุณ")

            # ตรวจสอบปัจจัยเสี่ยงแต่ละอย่าง
            risk_factors = []

            if glucose and glucose > 140:
                risk_factors.append("• ระดับน้ำตาลในเลือดสูง")
            if bmi > 25:
                risk_factors.append("• น้ำหนักเกินหรืออ้วน")
            if blood_pressure and blood_pressure > 90:
                risk_factors.append("• ความดันโลหิตสูง")
            if age > 45:
                risk_factors.append("• อายุมากกว่า 45 ปี")
            if diabetes_pedigree > 0.5:
                risk_factors.append("• มีประวัติเบาหวานในครอบครัว")

            if risk_factors:
                for factor in risk_factors:
                    st.markdown(factor)
            else:
                st.markdown("• ไม่พบปัจจัยเสี่ยงที่เด่นชัด")

            # แสดงคำแนะนำเพิ่มเติม
            st.markdown("#### การดูแลตัวเองเบื้องต้น")
            st.markdown("""
            1. ตรวจสุขภาพประจำปี
            2. หากมีอาการผิดปกติ เช่น ปัสสาวะบ่อย กระหายน้ำมาก อ่อนเพลียผิดปกติ ควรปรึกษาแพทย์
            3. รักษาน้ำหนักให้อยู่ในเกณฑ์ที่เหมาะสม
            """)

    # ส่วนท้ายของแอพ
    st.markdown("---")
    st.caption("⚠️ คำเตือน: ระบบนี้ใช้สำหรับการคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์")