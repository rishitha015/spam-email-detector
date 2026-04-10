import joblib

model = joblib.load("../model/spam_model.pkl")

# Take input from user
msg = input("Enter your message: ")

prediction = model.predict([msg])

if prediction[0] == 1:
    print("SPAM 🚫")
else:
    print("NOT SPAM ✅")
