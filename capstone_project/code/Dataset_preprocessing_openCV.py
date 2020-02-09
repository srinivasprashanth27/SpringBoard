char_list = string.ascii_letters + string.digits



def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return dig_lst




path = './mnt/ramdisk/max/90kDICT32px'


li_dirnames = []
li_filenames = []
# lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

# lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []




max_label_len = 0

i = 1
flag = 0

for root, dirnames, filenames in tqdm(os.walk(path)):
    li_dirnames.append(dirnames)
    li_filenames.append(filenames)
    for f_name in fnmatch.filter(filenames, '*.jpg'):
        # read input image and convert into gray scale image
        img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)

        # convert each image of shape (32, 128, 1)
        w, h = img.shape
        if h > 128 or w > 32:
            continue
        if w < 32:
            add_zeros = np.ones((32 - w, h)) * 255
            img = np.concatenate((img, add_zeros))

        if h < 128:
            add_zeros = np.ones((32, 128 - h)) * 255
            img = np.concatenate((img, add_zeros), axis=1)
        img = np.expand_dims(img, axis=2)

        # Normalize each image
        img = img / 255.

        # get the text from the image
        txt = f_name.split('_')[1]

        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)

        # split the 200000 data into validation and training dataset as 10% and 90% respectively
        if i % 10 == 0:
            valid_orig_txt.append(txt)
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)
            train_label_length.append(len(txt))
            train_input_length.append(31)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt))

            # break the loop if total data is 150000
        if(i%100==0):
            print('{} images processed'.format(i))

        if i == 200000:
            flag = 1
            break
        i += 1
    if flag == 1:
        break

# pad each output label to maximum text length



print("Storing the values to Array...")
print('Storing the data to local disk..')
np.save('training_img',training_img)
np.save('training_txt',training_txt)
np.save('train_input_length',train_input_length)
np.save('train_label_length',train_label_length)
np.save('orig_txt',orig_txt)
np.save('valid_img',valid_img)
np.save('valid_txt',valid_txt)
np.save('valid_input_length',valid_input_length)
np.save('valid_label_length',valid_label_length)
np.save('valid_orig_txt',valid_orig_txt)





print('Training Set Stats...')
print('Train image',len(training_img))
print('Train text',len(training_txt))
print('Validation data Stats...')
print('valid_img',len(valid_img))
print('Valid Text',len(valid_txt))
print('Max Label_length',max_label_len)
