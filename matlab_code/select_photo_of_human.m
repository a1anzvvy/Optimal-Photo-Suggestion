load AVA_with_human.mat

if exist('featOfPos.mat', 'file')
    load featOfPos.mat  %contain current information of processed image
else
    currIndex = 1;      %row index in selected matrix
    imageIndices = [];  %images that have been processed
    imageSize = [];     %image size on each row, [height, width]
    rectLT = [];        %rectangle left top point
    rectSize = [];      %rectangle size in the style of image size
%     save('featOfPos.mat', 'currIndex', 'imageIndices', 'imageSize', 'rectLT', 'rectSize')
end

initFlag = 0;
if initFlag
    currIndex = 1;      %row index in selected matrix
    imageIndices = [];  %images that have been processed
    imageSize = [];     %image size on each row, [height, width]
    rectLT = [];        %rectangle left top point
    rectSize = [];      %rectangle size in the style of image size
%     save('featOfPos.mat', 'currIndex', 'imageIndices', 'imageSize', 'rectLT', 'rectSize')
end

imageName = selected(:, 2);

while currIndex <= length(imageName)
    currImageName = imageName(currIndex);
    fileName = strcat('../images/', num2str(currImageName), '.jpg');
    image = imread(fileName);
    imshow(image);
    key = getkey();
    if key == 13
        imageIndices = [imageIndices; currImageName];
 
        shape = size(image);
        if length(shape) < 3
            shape = [shape, 1];
        end
        imageSize = [imageSize; shape];
        
        mouse = imrect;
        position = wait(mouse)
        rectLT = [rectLT; position(1:2)];
        rectSize = [rectSize; position(3), position(4)];
        
        currIndex = currIndex + 1;
%         save('featOfPos.mat', 'currIndex', 'imageIndices', 'imageSize', 'rectLT', 'rectSize');
    elseif key == 28 | key == 30
        currIndex = currIndex - 1;
    elseif key == 29 | key == 31
        currIndex = currIndex + 1;
    elseif key == 27
        break;
    end
end
% save('featOfPos.mat', 'currIndex', 'imageIndices', 'imageSize', 'rectLT', 'rectSize');
