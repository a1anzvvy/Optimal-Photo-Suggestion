function AVA_tagged = get_AVA_tag(tag1, tag2)
%output part of AVA with argumented tags    
AVA = load('../AVA_dataset/AVA.txt');
if nargin < 2
    index = AVA(:, 13) == tag1 | AVA(:, 14) == tag1;
    AVA_tagged = AVA(index, :);
    return
else
    index = (AVA(:, 13) == tag1 & AVA(:, 14) == tag2) | (AVA(:, 13) == tag2 & AVA(:, 14) == tag1);
    AVA_tagged = AVA(index, :);
end
end