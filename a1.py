from PIL import Image 

import imagehash

def compare_phash(img_path1,img_path2):
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        resized_img1 = img1.resize((32,32),resample=Image.LANCZOS)
        resized_img2 = img2.resize((32,32),resample=Image.LANCZOS)
        gray_img1 = resized_img1.convert("L")
        gray_img2= resized_img2.convert("L")
        hash1 = imagehash.phash(gray_img1)
        hash2 = imagehash.phash(gray_img2)
        print(hash1)
        print(hash2)
        distance = hash1 - hash2
        return distance

print("Differnce: ",compare_phash("D:\\coding\\Python\\pillow\\m1.jpg","D:\\coding\\Python\\pillow\\m4.jpg"))
  
 