function image_indices = copy_to(image_indices, shape, des)
%copy reshaped images to destination directory 
index = zeros(size(image_indices, 1), 1);
for i = 1 : size(image_indices, 1)
    image_ID = num2str(image_indices(i));
    filename = strcat('../images/', image_ID, '.jpg');
    if exist(filename, 'file')
        I = imread(filename);
        if length(size(I)) == 3 && size(I, 1) >= shape(1) && size(I, 2) >= shape(2)
            I = imresize(I, shape);
            des_file = strcat(des, '/', image_ID, '.jpg');
            imwrite(I, des_file);
            index(i) = 1;
        end
    end
end
image_indices = image_indices(logical(index));
end