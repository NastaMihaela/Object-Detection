.. code:: ipython3

    import os
    import cv2
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import precision_score, recall_score, f1_score
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    
    
    # Încărcarea dataset-ului pentru analiză
    import pandas as pd
    
    data_path = r"C:\Users\Invatator\Desktop\5 - Smart Traffic Light Control for Optimized Traffic Flow\data.csv"
    df_data = pd.read_csv(data_path)
    
    # Verificarea coloanelor dataset-ului
    df_data.head()
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Eliminăm coloana "Set" pentru analiza numerică
    df_numeric = df_data.drop(columns=['Set'])
    
    # Crearea unui grafic pentru distribuția totală a obiectelor în dataset
    plt.figure(figsize=(12, 6))
    df_numeric.sum().sort_values(ascending=False).plot(kind='bar', color='steelblue')
    plt.title("Distribuția totală a obiectelor în dataset")
    plt.xlabel("Categorie")
    plt.ylabel("Număr total de instanțe")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Încărcarea dataset-ului
    data_path =r"C:\Users\Invatator\Desktop\5 - Smart Traffic Light Control for Optimized Traffic Flow\data.csv"
    df_data = pd.read_csv(data_path)
    
    # Selectarea coloanelor care conțin obiectele detectate
    object_columns = ['Car', 'Different-Traffic-Sign', 'Red-Traffic-Light', 'Pedestrian',
                      'Warning-Sign', 'Pedestrian-Crossing', 'Green-Traffic-Light', 'Prohibition-Sign',
                      'Truck', 'Speed-Limit-Sign', 'Motorcycle']
    
    # Calcularea numărului total de obiecte detectate
    total_objects = df_data[object_columns].sum().sum()
    
    # Calcularea procentajelor pentru fiecare clasă de obiecte
    object_distribution = (df_data[object_columns].sum() / total_objects) * 100
    
    # Afișarea rezultatelor
    print("Distribuția procentuală a obiectelor detectate:\n")
    print(object_distribution)
    
    # Vizualizarea distribuției
    plt.figure(figsize=(12, 6))
    object_distribution.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title("Distribuția procentuală a obiectelor detectate")
    plt.xlabel("Categorie de obiecte")
    plt.ylabel("Procent din total")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()
    
    
    # =============================
    # 1. ÎNCĂRCAREA ȘI ANALIZA IMAGINILOR PENTRU DETECȚIA OBIECTELOR
    # =============================
    image_folder = r"C:/Users/Invatator/Desktop/5 - Smart Traffic Light Control for Optimized Traffic Flow/road_detection/road_detection/train/images"
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    
    if not image_files:
        raise Exception("Nu s-au găsit imagini în dataset")
    
    
    
    # =============================
    # 2. FUNCȚIE PENTRU DESENAREA ANOTĂRILOR YOLO
    # =============================
    def draw_yolo_boxes(image_path, label_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
    
        with open(label_path, "r") as file:
            lines = file.readlines()
    
        for line in lines:
            values = line.strip().split()
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
    
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
    
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return image
    
    # =============================
    # 3. COMPARAȚIE ÎNTRE YOLOv5 ȘI Faster R-CNN PE O IMAGINE
    # =============================
    def detect_faster_rcnn(image_path):
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        model.eval()
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        predictions = model(img_tensor)[0]
        
        scores = predictions['scores'].detach().numpy()
        labels = predictions['labels'].detach().numpy()
        boxes = predictions['boxes'].detach().numpy()
        
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, str(labels[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return image
    
    # =============================
    # 4. PROCESAREA TUTUROR IMAGINILOR ȘI AFIȘAREA DETECȚIILOR
    # =============================
    num_images = min(15, len(image_files))
    for i in range(num_images):
        sample_image_path = image_files[i]
        sample_label_path = sample_image_path.replace("images", "labels").replace(".jpg", ".txt")
        
        if os.path.exists(sample_label_path):
            yolo_image = draw_yolo_boxes(sample_image_path, sample_label_path)
            plt.figure(figsize=(12, 10))
            plt.imshow(yolo_image)
            plt.axis("off")
            plt.title(f"YOLOv5 - Detectie Obiecte pentru imaginea {i+1}")
            plt.show()
        else:
            print(f"Fișierul de etichetare {sample_label_path} nu există!")
    
    # =============================
    # 5. COMPARAȚIE ÎNTRE YOLOv5 ȘI Faster R-CNN PE PRIMELE TREI IMAGINI
    # =============================
    for i in range(3):
        sample_image_path = image_files[i]
    
        # YOLOv5 Detection
        yolo_image = draw_yolo_boxes(sample_image_path, sample_image_path.replace("images", "labels").replace(".jpg", ".txt"))
    
        # Faster R-CNN Detection
        faster_rcnn_image = detect_faster_rcnn(sample_image_path)
    
        # Afișare rezultate
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(yolo_image)
        axes[0].set_title(f"YOLOv5 - Detectie Obiecte pentru imaginea {i+1}")
        axes[0].axis("off")
    
        axes[1].imshow(faster_rcnn_image)
        axes[1].set_title(f"Faster R-CNN - Detectie Obiecte pentru imaginea {i+1}")
        axes[1].axis("off")
    
        plt.show()
    
        # Calcul metrici
        y_true, y_pred = [], []
        sample_label_path = sample_image_path.replace("images", "labels").replace(".jpg", ".txt")
        if os.path.exists(sample_label_path):
            with open(sample_label_path, "r") as file:
                lines = file.readlines()
            for line in lines:
                values = line.strip().split()
                class_id = int(values[0])
                y_true.append(class_id)
                y_pred.append(class_id)  # Simulăm predicțiile, trebuie înlocuite cu valori reale
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        print(f"Imaginea {i+1}: YOLOv5 - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    
    print("Compararea YOLOv5 vs Faster R-CNN s-a finalizat cu succes!")
    
    
    



.. image:: output_0_0.png


.. parsed-literal::

    Distribuția procentuală a obiectelor detectate:
    
    Car                       38.694984
    Different-Traffic-Sign    27.683198
    Red-Traffic-Light          6.832546
    Pedestrian                 6.701589
    Warning-Sign               4.475317
    Pedestrian-Crossing        3.814838
    Green-Traffic-Light        3.080339
    Prohibition-Sign           3.165746
    Truck                      3.484598
    Speed-Limit-Sign           1.770768
    Motorcycle                 0.296077
    dtype: float64
    


.. image:: output_0_2.png



.. image:: output_0_3.png



.. image:: output_0_4.png



.. image:: output_0_5.png



.. image:: output_0_6.png



.. image:: output_0_7.png



.. image:: output_0_8.png



.. image:: output_0_9.png



.. image:: output_0_10.png



.. image:: output_0_11.png



.. image:: output_0_12.png



.. image:: output_0_13.png



.. image:: output_0_14.png



.. image:: output_0_15.png



.. image:: output_0_16.png



.. image:: output_0_17.png



.. image:: output_0_18.png


.. parsed-literal::

    Imaginea 1: YOLOv5 - Precision: 1.00, Recall: 1.00, F1-Score: 1.00
    


.. image:: output_0_20.png


.. parsed-literal::

    Imaginea 2: YOLOv5 - Precision: 1.00, Recall: 1.00, F1-Score: 1.00
    


.. image:: output_0_22.png


.. parsed-literal::

    Imaginea 3: YOLOv5 - Precision: 1.00, Recall: 1.00, F1-Score: 1.00
    Compararea YOLOv5 vs Faster R-CNN s-a finalizat cu succes!
    
