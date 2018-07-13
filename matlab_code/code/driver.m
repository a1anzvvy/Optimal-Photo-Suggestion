tag1 = 35;
AVA_tagged = get_AVA_tag(tag1);
shape = [320, 320];
des = '../demo1';
image_info = copy_to_des_resize(AVA_tagged, shape, des);
label = generate_label(image_info);
dlmwrite('../demo1/label.txt', label, 'delimiter', ' ', 'precision', '%.8f')