from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from io import BytesIO
import os
import base64

# 画像データのBase64エンコード
def get_image_base64(img_file):
    img_file.seek(0)
    img_data = base64.b64encode(img_file.read()).decode('utf-8')
    img_data = 'data:image/jpeg;base64,' + img_data
    return img_data

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        # POSTリクエストによるアクセス時の処理を記述
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            # 4章で、画像ファイル（img_file）の前処理を追加
            img_file = BytesIO(img_file.read())
            img_file.seek(0)
            img_data = get_image_base64(img_file)

            img_file.seek(0)
            img = load_img(img_file, target_size=(224, 224))#VGG16_targetsize=224×224
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))#任意のサンプル数×224×224×3

            # VGG16の入力形式に前処理
            img_array = preprocess_input(img_array)

            # モデルを読み込み、予測を行い、結果をテンプレートに渡す
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            preds = model.predict(img_array)
            top_preds = decode_predictions(preds, top=5)[0]

            # 画像と予測結果をテンプレートに渡す
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': top_preds, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})

def get_image_base64(img_file):
    img_file.seek(0)
    img_data = base64.b64encode(img_file.read()).decode('utf-8')
    return f'data:image/jpeg;base64,{img_data}'

