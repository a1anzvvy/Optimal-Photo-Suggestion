function label = generate_label(image_info)
label = zeros(size(image_info, 1), 3);
label(:, 1) = image_info(:, 2);
score_range = 1 : 10;
number = image_info(:, 3 : 12);
score = sum(number .* score_range, 2) ./ sum(number, 2);
label(:, 2) = score > 5.5;
label(:, 3) = 1 - label(:, 2);

end