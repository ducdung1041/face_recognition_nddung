from flask import Flask, render_template, redirect, url_for, send_from_directory
import os
import cv2
import face_recognition
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import pickle
from sklearn import preprocessing

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nddung'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

# Load the model from the file
load_model = pickle.load(open('D:\\face\\face_recognition_model5_cnn.pkl', 'rb'))

# Tải các vectơ mã hóa khuôn mặt và nhãn từ file
def load_encoded_faces(filename):
    with open(filename, 'rb') as f:
        loaded_images = pickle.load(f)
        loaded_labels = pickle.load(f)
    return loaded_images, loaded_labels

# Hàm padding hình ảnh
def pad_image(image, desired_size):
    old_size = image.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

# Tải các vectơ mã hóa khuôn mặt và nhãn từ file
loaded_images, loaded_labels = load_encoded_faces('D:\\face\\encoded_faces_cnn .pkl')

# Tạo một bộ mã hóa nhãn và fit với các nhãn hiện có
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(loaded_labels)

# Chuyển đổi các nhãn thành số nguyên
encoded_labels = label_encoder.transform(loaded_labels)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)

        # Process the image
        image_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        image = cv2.imread(image_path)
        face_locations = face_recognition.face_locations(image, model="hog")

        for face_location in face_locations:
            top, right, bottom, left = face_location
            # image = pad_image(image, max(image.shape))
            # image = cv2.resize(image, (350, 350))
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            face_encodings = face_recognition.face_encodings(image, [face_location])
            for face_encoding in face_encodings:
                face_encoding = face_encoding.reshape(1, -1)
                predicted_label = load_model.predict(face_encoding)[0]
                confidence = max(load_model.predict_proba(face_encoding)[0])*100
                predicted_label = [predicted_label]
                # In nhãn và tỷ lệ phần trăm tương ứng
                if confidence > 60:  # Thay đổi ngưỡng này nếu cần
                    predicted_name = label_encoder.inverse_transform(predicted_label)[0]
                    cv2.putText(image, f"{predicted_name} ({confidence:.2f}%)", (left, bottom+15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                else:
                    cv2.putText(image, "Unknown", (left, bottom+15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Save the image with rectangles
        processed_filename = 'processed_' + filename
        processed_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], processed_filename)
        cv2.imwrite(processed_path, image)
        return redirect(url_for('get_file', filename=processed_filename))
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url)

if __name__ == "__main__":
    app.run(debug=True)
