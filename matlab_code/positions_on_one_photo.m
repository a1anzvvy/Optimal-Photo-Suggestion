%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileName = 'mountain.jpg';%%% remember to have a valid file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I = imread(fileName);
imshow(I);
height = size(I, 1);
width = size(I, 2);

corners   = detectFASTFeatures(rgb2gray(I));
strongest = selectStrongest(corners,200);
[hog2, validPoints, ~] = extractHOGFeatures(I,strongest);

hog2 = hog2(1 : 50, :);
cat1 = validPoints.Location(1 : 50, 1) / width;
cat2 = validPoints.Location(1 : 50, 2) / height;

HOGfeature = [hog2, cat1, cat2];
HOGfeature = reshape(HOGfeature', 1, []);

positionArray = [];
featureSpace = [];
for i = 1:10
    mouse = imrect;
    position = wait(mouse);
    
    positionAndScale = [position(1) / width, position(2) / height, position(3) / width, position(4) / height];
    feature = [HOGfeature, positionAndScale];
    
    positionArray = [positionArray; position];
    featureSpace = [featureSpace; feature];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I = imresize(I, [320, 320]);
imwrite(I, '../test_score_distribution/1111111.jpg');
dataToWrite = [repmat(1111111, size(featureSpace, 1)),featureSpace, positionArray];
dlmwrite('../test_score_distribution/test.txt', dataToWrite, 'delimiter', ' ', 'precision', '%.8f');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%