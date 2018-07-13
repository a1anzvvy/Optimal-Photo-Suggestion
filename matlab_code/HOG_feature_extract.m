load AVA_with_human
load human_position_scale

%% delete not rgb image
index = imageSize(:, 3) == 1;
imageIndices(index, :) = [];
imageSize(index, :) = [];
rectLT(index, :) = [];
rectSize(index,:) = [];

%% delete image with greater height
% index = imageSize(:, 1) > imageSize(:, 2);
% imageIndices(index, :) = [];
% imageSize(index, :) = [];
% rectLT(index, :) = [];
% rectSize(index,:) = [];
% 
% clear index

%% delete image smaller than assigned shape
shape = [320, 320];
index = imageSize(:, 1) < shape(1) | imageSize(:, 1)< shape(2);
imageIndices(index, :) = [];
imageSize(index, :) = [];
rectLT(index, :) = [];
rectSize(index,:) = [];

clear index

%% position and scale information
valid_index = ones(size(imageIndices, 1), 1);
ind = 1;
imageName = imageIndices;
featureSpace = [];
while ind <= length(imageIndices)
    %read image
    currImageName = imageName(ind);
    fileName = strcat('../images/', num2str(currImageName), '.jpg');    
    I2 = imread(fileName);
    %build hog feature and keypoint location array 
    height = imageSize(ind, 1);
    width = imageSize(ind, 2);
    corners   = detectFASTFeatures(rgb2gray(I2));
    strongest = selectStrongest(corners, 300);
    [hog2, validPoints, ~] = extractHOGFeatures(I2, strongest);
    if size(hog2, 1) < 50
        valid_index(ind) = 0;
        ind = ind + 1;
        continue
    end
    hog2 = hog2(1 : 50, :);
    cat1 = validPoints.Location(1 : 50, 1) / width;
    cat2 = validPoints.Location(1 : 50, 2) / height;
    feature = [hog2, cat1, cat2];
    feature = reshape(feature', 1, []);
    %human position and scale in the image.
    posAndScale = [rectLT(ind, 1) / width, rectLT(ind, 2) / height, ...
                    rectSize(ind, 1) / width, rectSize(ind, 2) / height];
    feature = [feature, posAndScale];
    featureSpace = [featureSpace; feature];
    ind = ind + 1;
end
imageIndices = imageIndices(logical(valid_index));
info = [imageIndices, featureSpace];
dlmwrite('../demo/info.txt', info, 'delimiter', ' ', 'precision','%.8f');
copy_to(imageIndices, shape, '../demo');

image_info = [];
for i = 1: length(imageIndices)
    image_info = [image_info; selected(selected(:, 2) == imageIndices(i), :)];
end

label = generate_label(image_info);

dlmwrite('../demo/label.txt', label(:, 2:3), 'delimiter', ' ', 'precision','%.8f');

