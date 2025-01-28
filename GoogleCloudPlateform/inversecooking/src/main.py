import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from utils.output_utils import prepare_output
from PIL import Image
import time
import requests
from io import BytesIO
import random
from collections import Counter
from flask import Flask, request, jsonify
import io


def detect(input_image):

    #data_dir = './data'

    # code will run in gpu if available and if the flag is set to True, else it will run on cpu
    use_gpu = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

    # code below was used to save vocab files so that they can be loaded without Vocabulary class
    #ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'final_recipe1m_vocab_ingrs.pkl'), 'rb'))
    #ingrs_vocab = [min(w, key=len) if not isinstance(w, str) else w for w in ingrs_vocab.idx2word.values()]
    #vocab = pickle.load(open(os.path.join(data_dir, 'final_recipe1m_vocab_toks.pkl'), 'rb')).idx2word
    #pickle.dump(ingrs_vocab, open('../demo/ingr_vocab.pkl', 'wb'))
    #pickle.dump(vocab, open('../demo/instr_vocab.pkl', 'wb'))

    ingrs_vocab = open('ingr_vocab.txt', 'r')
    vocab = open('instr_vocab.txt', 'r')

    ingrs_vocab = ingrs_vocab.read().splitlines()
    vocab = vocab.read().splitlines()
    
    
    
    ingr_vocab_size = len(ingrs_vocab)
    instrs_vocab_size = len(vocab)
    output_dim = instrs_vocab_size

    print (instrs_vocab_size, ingr_vocab_size)


    t = time.time()
    import sys; sys.argv=['']; del sys
    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only=False
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
    # Load the trained model parameters
    model_path = 'modelbest.ckpt'
    model.load_state_dict(torch.load(model_path, map_location=map_loc))
    model.to(device)
    model.eval()
    model.ingrs_only = False
    model.recipe_only = False
    print ('loaded model')
    print ("Elapsed time:", time.time() -t)

    transf_list_batch = []
    transf_list_batch.append(transforms.ToTensor())
    transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), 
                                                (0.229, 0.224, 0.225)))
    to_input_transf = transforms.Compose(transf_list_batch)

    greedy = [True]
    beam = [-1]

    temperature = 1.0
    numgens = len(greedy)

    use_urls = False # set to true to load images from demo_urls instead of those in test_imgs folder
    show_anyways = False #if True, it will show the recipe even if it's not valid
    # image_folder = os.path.join(data_dir, 'demo_imgs')

    # if not use_urls:
    #     demo_imgs = os.listdir(image_folder)
    #     random.shuffle(demo_imgs)

    # demo_urls = ['https://food.fnr.sndimg.com/content/dam/images/food/fullset/2013/12/9/0/FNK_Cheesecake_s4x3.jpg.rend.hgtvcom.826.620.suffix/1387411272847.jpeg',
    #             'https://www.196flavors.com/wp-content/uploads/2014/10/california-roll-3-FP.jpg']

    # demo_files = demo_urls if use_urls else demo_imgs
    save_res = None



    image = input_image
    
    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)
    
    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)
    
    plt.imshow(image_transf)
    plt.axis('off')
    plt.show()
    plt.close()
    
    num_valid = 1
    for i in range(numgens):
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=greedy[i], 
                                temperature=temperature, beam=beam[i], true_ingrs=None)
            
        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()
            
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
        
            
        num_valid+=1

        #print (', '.join(outs['ingrs']))
        save_res = outs['ingrs']
    #print(save_res)

    return save_res


# if __name__ == "__main__":

#     img = Image.open('/home/r10944006/GDSC/inversecooking/data/demo_imgs/1.jpg')
#     print(detect(img))

def read_file(filename):
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            lines.append(line.strip())
    save = []
    for _, i in enumerate(lines):
      save.append(i.split(" "))
    return save

def get_recognition(test):
    path_for_result = "ans.txt"

    res_arr = read_file(path_for_result)
    
    save_arr = []
    for _, i in enumerate(test):
      for _, j in enumerate(res_arr):
        if len(j) == 2:
          k, p = j[0], j[1]
          if (i == k):
            save_arr.append(j)
    return save_arr

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            np_image = np.asarray(pillow_img)
            prediction = detect(pillow_img)
            data = get_recognition(prediction)
            
            save = []
            for _, i in enumerate(data):
                save.append(i[0] +" "+ i[1])

            return jsonify({"result" : save})
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run( debug=True)