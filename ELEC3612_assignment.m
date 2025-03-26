% ELEC3612 Digital Media Engineering
% Assignment: Implementation of JPEG Encoding in MATLAB
% Harry Carless

% Clear workspace and close all figures
clear;
close all;

%% ==== STEP 1: READ AND DISPLAY IMAGE ==== %%

% Load Image and convert to grayscale
image = imread("lenna.png");
gray_image = rgb2gray(image);

% Display Original and Grayscale Images
figure('Name', 'Step 1: Image Loading');

subplot(1, 2, 1);
imshow(original_image);
title('Original RGB Image');

subplot(1, 2, 2);
imshow(gray_image);
title('Grayscale Image');

%% ==== STEP 2: Divide Image into 8x8 Blocks === %%

% Extract Image Dimensions (512 x 512) and Define Block Dimensions
[rows, cols] = size(gray_image);
block_size = 8;

% Number of Blocks per row and column
num_blocks_row = row / block_size;
num_blocks_col = cols / block_size;

% Array to temp store the blocks
blocks_cell = cell(1, total_blocks);
block_idx = 1;
for i = 1:block_size:rows
    for j = 1:block_size:cols
        blocks_cell{block_idx} = gray_image(i:i+block_size-1, j:j+block_size-1);
        block_idx = block_idx + 1;
    end
end

% Convert cell array to a 3D array
blocks = cat(3, blocks_cell{:});

%% ==== STEP 3: Apply Discrete Cosine Transform (DCT) ==== %%


