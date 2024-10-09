import streamlit as st
import pandas as pd
import joblib
import time

st.title("Welcome to the car price prediction!")

model = joblib.load("final_model.joblib")
pipeline = joblib.load("pre_pipeline.joblib")
necessary_columns = joblib.load("necessary_columns.joblib")

marka_list = ["--", 'MG', 'Opel', 'Toyota', 'Mercedes', 'BMW', 'LADA (VAZ)', 'Hyundai',
       'Chevrolet', 'Fiat', 'Peugeot', 'Honda', 'KamAz', 'Kia',
       'Land Rover', 'Jeep', 'Lexus', 'Ford', 'Hongqi', 'Audi', 'GAC',
       'BYD', 'Volkswagen', 'Mitsubishi', 'Renault', 'Avatr', 'Changan',
       'ZEEKR', 'Mazda', 'GAZ', 'Nissan', 'MAN', 'Subaru', 'Voyah',
       'Bestune', 'DAF', 'Maple', 'Daewoo', 'Chery', 'Maserati', 'Volvo',
       'Baic', 'Wuling', 'Porsche', 'Infiniti', 'Suzuki', 'Khazar', 'GMC',
       'Haval', 'Li Auto', 'ROX (Polar Stone)', 'Mini', 'Smart', 'Tesla',
       'BMW Alpina', 'Saipa', 'Geely', 'Ssang Yong', 'Acura', 'Genesis',
       'Lynk & Co', 'Skoda', 'Dodge', 'Dayun', 'Iran Khodro', 'Forthing',
       'GWM', 'Isuzu', 'Citroen', 'Cadillac', 'Scania', 'Neta', 'DFSK',
       'VGV', 'Jaguar', 'DongFeng', 'ZX Auto', 'Tofas', 'SEAT', 'UAZ',
       'HOWO', 'KrAZ', 'XPeng', 'Leapmotor', 'Alfa Romeo', 'Ravon',
       'Abarth', 'JAC', 'Karry', 'IM', 'FAW', 'KG Mobility', 'Jetour',
       'Nio', 'JMC', 'Dacia', 'Haima', 'Xiaomi', 'ZIL', 'Renault Samsung',
       'KAIYI', 'Lincoln', 'Chrysler', 'Foton']
marka = st.selectbox("Markanı seçin: ",
                     (marka_list))
if marka == "--":
    marka = str(" ")

oturucu = st.selectbox(("Ötürücü tipini seçin: "),
                       ("--", "Ön", "Arxa", "Tam"))
if oturucu == "--":
    oturucu = str(" ")

suretler_qutusu = st.selectbox(("Sürətlər qutusu tipini seçin: "),
                            ("--", 'Avtomat', 'Mexaniki', 'Variator', 'Reduktor', 'Robot'))
if suretler_qutusu == "--":
    suretler_qutusu = str(" ")

yanacaq = st.selectbox("Yanacaq növünü seçin:", 
                       ("--", 'Benzin', 'Hibrid', 'Dizel', 'Plug-in Hibrid', 'Elektro', 'Qaz'))
if yanacaq == "--":
    yanacaq = str(" ")

ban_novleri = ['--', 'Sedan', 'Hetçbek', 'Liftbek', 'Offroader ', 'Furqon', 'SUV Kupe', 'Yük maşını', 'Minivan', 'Universal',
               'Pikap, ikiqat kabin', 'Kompakt-Van', 'Dartqı', 'Kabriolet', 'Kupe', 'Mikroavtobus', 'Hetçbek, 4 qapı', 
               'Pikap, tək kabin', 'Avtobus', 'Rodster']
ban_novu = st.selectbox("Ban növünü seçin: ",
                          (ban_novleri))
if ban_novu == "--":
    ban_novu = " "

option_deri = st.selectbox(("Dəri salon: "),
                           ("--", "Var", "Yox"))

if option_deri == "Var":
    deri_salon = True
else:
    deri_salon = False

option_ventil = st.selectbox(("Oturacaqların ventilyasiyası: "),
                           ("--", "Var", "Yox"))

if option_ventil == "Var":
    ventilyasiya = True
else:
    ventilyasiya = False

option_lyuk = st.selectbox(("Lyuk: "),
                           ("--", "Var", "Yox"))

if option_lyuk == "Var":
    lyuk = True
else:
    lyuk = False

option_salon = st.selectbox(("Avtosalon: "),
                           ("--", "Bəli", "Xeyr"))

if option_salon == "Bəli":
    salon = str("Yes")
else:
    salon = str("No")

year = st.text_input("Buraxılış ili: ", 0)
year = int(year)

yurus = st.text_input("Yürüş (km): ", 0)
yurus = int(yurus)

motor = st.text_input("Motor: ", 0)
motor = float(motor)

at_gucu = st.text_input("At gücü: ", 0)
at_gucu = int(at_gucu)

datapoint = {"Marka": [marka],
             "Ötürücü": [oturucu],
             "Sürətlər qutusu": [suretler_qutusu],
             "Ban növü": [ban_novu],
             "Dəri salon": [deri_salon],
             "Yanacaq": [yanacaq],
             "Oturacaqların ventilyasiyası": [ventilyasiya],
             "Avtosalon": [salon],
             "Lyuk": [lyuk],
             "Buraxılış ili": [year],
             "Yürüş": [yurus],
             "Motor": [motor],
             "At gücü": [at_gucu]}

datapoint_df = pd.DataFrame(datapoint)

cat_col = list(datapoint_df.select_dtypes(include=["object", "bool"]).columns)
num_col = list(datapoint_df.select_dtypes(exclude=["object", "bool"]).columns)

end_point = pipeline.transform(datapoint_df)

categorical_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
onehot_encoded_cols = categorical_encoder.get_feature_names_out(cat_col)

# Combine the numerical columns with the one-hot encoded column names
all_feature_names = list(num_col) + list(onehot_encoded_cols)

normal_endpoint = pd.DataFrame(end_point, columns=all_feature_names)

final_endpoint = normal_endpoint[necessary_columns]

if st.button("Qiymət təxmini: "):
    progress_text = "Hesablanır . . . "
    my_bar = st.sidebar.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete+1, text=progress_text)
    my_bar.empty()

    price = float(model.predict(final_endpoint)[0])
    st.write(f"Qiymət: {round(price)} AZN")

