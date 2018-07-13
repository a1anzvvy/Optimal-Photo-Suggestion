function image_info = copy_to_des_resizing(AVA_tagged, shape, des)
% copy resized image to destination directory
% des is relative dir to current dir
index = zeros(size(AVA_tagged, 1), 1);
for i = 1 : size(AVA_tagged, 1)
    image_ID = num2str(AVA_tagged(i, 2));
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
image_info = AVA_tagged(logical(index), :);
end