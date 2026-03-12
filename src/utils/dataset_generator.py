import argparse
import os
import yaml
import sys
from pathlib import Path
from colorama import Fore

# Proje kök dizinini sys.path'e ekleyelim
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataloader import Dataloader
from src.generator import PaddleOCRGenerator # Varsayılan olarak Paddle kullanıyoruz

class CustomPathDataloader(Dataloader):
    """
    Dataloader'ı ezerek istediğimiz herhangi bir yoldan veri okumasını sağlıyoruz.
    """
    def __init__(self, dataset_path: str, dictionary: list[str] | None):
        self.custom_path = dataset_path
        # Dataloader.__init__ çağırmıyoruz çünkü o 'data' klasörüne bakıyor
        self.datasets = ["custom"]
        self.dict = set(dictionary) if dictionary is not None else None
        self.data = {"train": [], "test": []}
        self.workers = os.cpu_count()
        self.load_data()

    def load_data(self) -> None:
        from src.utils import reader
        import multiprocessing
        
        # Kullanıcının verdiği yolda 'labels' klasörü var mı kontrol et
        labels_path = os.path.join(self.custom_path, "labels")
        if not os.path.exists(labels_path):
            # Eğer labels klasörü yoksa direkt klasörün içine bakmayı deneyelim
            labels_path = self.custom_path
            
        full_paths = [os.path.join(labels_path, fn) for fn in os.listdir(labels_path) if fn.endswith(".txt")]
        
        if not full_paths:
            print(f"{Fore.RED}✗ {labels_path} içinde .txt etiketi bulunamadı!{Fore.RESET}")
            return

        with multiprocessing.Pool(processes=self.workers) as pool:
            # Tüm veriyi 'train' olarak kabul ediyoruz kolaylık için
            self.data["train"] = self.filter(pool.map(reader.read_label, full_paths))
            print(f"{Fore.GREEN}✓ {len(self.data['train'])} etiket başarıyla yüklendi.{Fore.RESET}")

def start_custom_pipeline(dataset_path: str, output_name: str):
    # Mevcut pipeline.yaml'dan genel ayarları alalım (augmentation vb.)
    config_path = os.path.join(project_root, "pipeline.yaml")
    with open(config_path, "r") as f:
        pipeline_config = yaml.safe_load(f)

    # Sözlük yolunu düzelt
    dict_path = os.path.join(project_root, pipeline_config.get("dict", "./dict/en_dict.txt"))
    dictionary = None
    if os.path.exists(dict_path):
        dictionary = [line.strip() for line in open(dict_path, "r").readlines()]

    print(f"{Fore.LIGHTCYAN_EX}--- Custom Dataset Generator ---{Fore.RESET}")
    print(f"📁 Dataset: {dataset_path}")
    print(f"📦 Output: {output_name}")

    # 1. Custom Dataloader oluştur
    dataloader = CustomPathDataloader(dataset_path, dictionary)

    # 2. Generator'ı hazırla
    generator = PaddleOCRGenerator(
        test_name=output_name,
        datasets=["custom"],
        dict=dictionary,
        workers=pipeline_config.get("workers", 4),
        augmentation=pipeline_config.get("augmentation", True),
        augmentation_config=pipeline_config.get("augmentation-config")
    )

    # 3. Veri üretimine başla
    tasks = pipeline_config.get("tasks", {"det": "y", "rec": "y"})
    
    # generate_data metodunu dataloader'ımızı kullanacak şekilde çağıralım
    # Generator.generate_data normalde kendi içinde dataloader yaratır, 
    # biz onu biraz bypass etmelisiniz veya dataloader'ı oraya vermelisiniz.
    # generator.generate_data(tasks) # Bu satır dataloder'ı yeniden yaratır.
    
    # Manuel olarak generate_data içindeki işlemleri yapıyoruz:
    generator.root_path = os.path.join(generator.base_path, f"{output_name}-{generator.name()}")
    os.makedirs(generator.root_path, exist_ok=True)
    
    print(f"\n{Fore.LIGHTCYAN_EX}[Generation Step]{Fore.RESET}")
    for task_name, active in tasks.items():
        if active == "y":
            print(f"⏳ Generating {task_name} data...")
            if task_name == "det":
                generator._generate(dataloader, "Detection")
            elif task_name == "rec":
                generator._generate(dataloader, "Recognition")
    
    print(f"\n{Fore.GREEN}✨ Pipeline başarıyla tamamlandı!{Fore.RESET}")
    print(f"📂 Sonuçlar burada: {generator.root_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Dataset Pipeline for Custom Paths")
    parser.add_argument("--dataset", required=True, help="İşlenecek veriseti klasörü yolu")
    parser.add_argument("--output", default="custom_x_dataset", help="Çıktı klasörü ismi")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"{Fore.RED}Hata: {args.dataset} yolu bulunamadı!{Fore.RESET}")
        sys.exit(1)
        
    start_custom_pipeline(args.dataset, args.output)
