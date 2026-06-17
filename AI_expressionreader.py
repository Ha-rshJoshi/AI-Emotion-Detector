import cv2
import random
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotions = [
    "Happy 😊",
    "Sad 😔",
    "Surprised 😲",
    "Neutral 😐",
    "Angry 😠",
    "Confident 😎"
]

responses = {
    "Happy 😊": ["Keep smiling, great things are happening!",
        "Your positive energy is inspiring.",
        "Looks like you're having a wonderful day.",
        "Happiness is the best accessory.",
        "Success starts with a positive mindset."],
    "Sad 😔": ["Every challenge is an opportunity to grow.",
        "Better days are ahead, stay strong.",
        "Take a deep breath and keep moving forward.",
        "It's okay to have tough days.",
        "Small steps still move you forward."],
    "Surprised 😲": ["Wow! That caught your attention.",
        "Looks like something unexpected happened.",
        "A moment worth remembering!",
        "That's an interesting reaction.",
        "Curiosity leads to discovery."],
    "Neutral 😐": ["Staying focused and composed.",
        "A calm mind makes better decisions.",
        "Balance is a strength.",
        "Maintaining a steady mindset.",
        "Focused and ready for the next task."],
    "Angry 😠": [
        "Take a moment to reset and refocus.",
        "Stay calm and think clearly.",
        "Patience is a powerful skill.",
        "Every problem has a solution.",
        "Control your response, not the situation."
],
    "Confident 😎": [
        "Confidence is the key to success.",
        "You look ready to take on challenges.",
        "Believe in your abilities.",
        "Strong mindset, strong results.",
        "Leadership starts with self-belief."
]
    }

cap = cv2.VideoCapture(0)

emotion = random.choice(emotions)
reply = random.choice(responses[emotion])

last_change = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if time.time() - last_change > 5:
        emotion = random.choice(emotions)
        reply = random.choice(responses[emotion])
        last_change = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        reply,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow("AI Expression Reader", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
