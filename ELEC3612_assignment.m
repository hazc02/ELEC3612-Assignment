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
imshow(image);
title('Original RGB Image');

subplot(1, 2, 2);
imshow(gray_image);
title('Grayscale Image');

%% ==== STEP 2: Divide Image into 8x8 Blocks === %%

% Extract Image Dimensions (512 x 512) and Define Block Dimensions
[rows, cols] = size(gray_image);
block_size = 8;

% Number of Blocks per row and column
num_blocks_row = rows / block_size;
num_blocks_col = cols / block_size;
total_blocks = num_blocks_col * num_blocks_row;

% Array to temporarily store the blocks
blocks_cell = cell(1, total_blocks);
block_idx = 1;

% Extract the block and center it around 0 by subtracting 128
for i = 1:block_size:rows
    for j = 1:block_size:cols
        block = double(gray_image(i:i+block_size-1, j:j+block_size-1)) - 128;
        blocks_cell{block_idx} = block;
        block_idx = block_idx + 1;
    end
end

% Convert cell array to a 3D array
blocks = cat(3, blocks_cell{:});
 

% Figure to display the grayscale image with an 8x8 block grid overlay
figure('Name', 'Step 2: 8x8 Blocks');
imshow(gray_image);
hold on;

% Draw vertical grid lines every 8 pixels
for x = block_size:block_size:cols
    line([x, x], [1, rows], 'Color', 'r', 'LineWidth', 0.5);
end

% Draw horizontal grid lines every 8 pixels
for y = block_size:block_size:rows
    line([1, cols], [y, y], 'Color', 'r', 'LineWidth', 0.5);
end

% Add labels and title
title('Grayscale Image with 8x8 Block Grid Overlay (JPEG Compression)');
xlabel('Columns');
ylabel('Rows');
hold off;

%% ==== STEP 3: Apply Discrete Cosine Transform (DCT) ==== %%

% Initialise array for DCT coefficients
dct_blocks = zeros(size(blocks)); 

% Apply DCT to each block
for k = 1:total_blocks
    dct_blocks(:,:,k) = dct2(blocks(:,:,k)); 
end

% Select one block for visualisation (1st block)
selected_dct_block = dct_blocks(:,:,1);

% Prepare the DCT coefficients for log-scaled display
% Use absolute values and a small offset to handle negatives and zeros
dct_display = abs(selected_dct_block) + 1e-5;

% Log-scale for better visibility with shading
log_dct = log10(dct_display); 

% Display the log-scaled coefficients as a heatmap with actual values overlaid
figure('Name', 'Step 3: DCT Coefficients with Actual Values');

% Generate heatmap
imagesc(log_dct); 

title('DCT Coefficients of 1st 8x8 Block (Centered Pixel Values)');
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');

% Add grid lines
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5); 
grid on;

% Overlay the actual (non-log-scaled) DCT coefficient values
hold on;
for row = 1:8
    for col = 1:8
        % Get the actual DCT value
        actual_value = selected_dct_block(row, col);

        % Display the actual value, rounded to 1 decimal place
        text(col, row, sprintf('%.1f', actual_value), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle', ...
             'Color', 'black', ...
             'FontSize', 8);
    end
end
hold off;

%% ==== STEP 4: Quantization ==== %% 

% Standard JPEG Luminance Quantization Matrix
jpeg_quant_matrix = [
    16 11 10 16 24 40 51 61;
    12 12 14 19 26 58 60 55;
    14 13 16 24 40 57 69 56;
    14 17 22 29 51 87 80 62;
    18 22 37 56 68 109 103 77;
    24 35 55 64 81 104 113 92;
    49 64 78 87 103 121 120 101;
    72 92 95 98 112 100 103 99
];

% Initialise array for quantized coefficients
quantized_blocks = zeros(size(dct_blocks)); 

% Quantize each block
for k = 1:total_blocks
    quantized_blocks(:,:,k) = round(dct_blocks(:,:,k) ./ jpeg_quant_matrix); 
end

%% ==== ZIGZAG ORDER FUNCTION ==== %%

function zigzag_order = generate_zigzag_order(block_size)
    % Purpose: Generate the zig-zag scanning order for an NxN block.
    % Input: block_size - Size of the block
    % Output: zigzag_order - Vector of indices in zig-zag order.
    
    N = block_size;

    % Initialise the order vector
    zigzag_order = zeros(1, N*N); 

    idx = 1; % Current position in the order vector
    row = 1; % Start at top-left
    col = 1;
    
    % Direction flag: true for up-right, false for down-left
    going_up = true; 
    
    while idx <= N*N
        % Add current position to the order
        zigzag_order(idx) = (row-1)*N + col;
        idx = idx + 1;
        
        if going_up
            % Moving up-right
            if col == N
                row = row + 1; % Hit right edge, move down
                going_up = false;
            elseif row == 1
                col = col + 1; % Hit top edge, move right
                going_up = false;
            else
                row = row - 1; % Move up
                col = col + 1; % Move right
            end
        else
            % Moving down-left
            if row == N
                col = col + 1; % Hit bottom edge, move right
                going_up = true;
            elseif col == 1
                row = row + 1; % Hit left edge, move down
                going_up = true;
            else
                row = row + 1; % Move down
                col = col - 1; % Move left
            end
        end
    end
end

%% ==== STEP 5: ZIG-ZAG Scanning ==== %%

% Generate zig-zag order using my own function
zigzag_order = generate_zigzag_order(block_size); 

% Initialise array for scanned vectors
scanned_vectors = zeros(total_blocks, 64); 

% Apply zig-zag scanning
for k = 1:total_blocks
    block = quantized_blocks(:,:,k);
    scanned_vectors(k, :) = block(zigzag_order); 
end

% Show the zig-zag scanned vector of the first block as a heatmap
% Reconstruct the 8x8 block with values in their original positions
display_block = zeros(block_size);
first_vector = scanned_vectors(1, :);

for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    display_block(row, col) = first_vector(idx);
end

% Create a heatmap with the zig-zag order overlaid
figure('Name', 'Step 5: Zig-Zag Scanned Coefficients of First Block');

imagesc(display_block); % Generate heatmap of the coefficients

title('Zig-Zag Scanned Coefficients of First 8x8 Block');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5); % Add grid lines
grid on;

% Format to overlay the zig-zag order indices on each cell
hold on;
for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    text(col, row, sprintf('%d', idx), 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', 'Color', 'Black', 'FontSize', 8);
end
hold off;

%% ==== RUN-LENGTH ENCODING FUNCTION ==== %%

function [values, counts] = rle_encode(vector)

    % Initalise values and count
    values = vector(1); 
    counts = 1; 

    for i = 2:length(vector)
        % If the current value is the same as the previous one, increment
        if vector(i) == vector(i-1) 
            counts(end) = counts(end) + 1; 
        else        
            values = [values, vector(i)];  
            counts = [counts, 1]; 
        end
    end
end

%% ==== STEP 6: Run-Length Encoding (RLE) ==== %%

% Array for RLE values and counts
rle_values = cell(1, total_blocks);
rle_counts = cell(1, total_blocks);

% Encode each block
for k = 1:total_blocks
    [values, counts] = rle_encode(scanned_vectors(k, :)); 
    rle_values{k} = values;
    rle_counts{k} = counts;
end

% Get the RLE output for the first block
values = rle_values{1};
counts = rle_counts{1};

% Create a figure to display the RLE output as a text-based table
figure('Name', 'Step 6: RLE Output of First Block', 'Position', [100, 100, 800, 200]);

% Format Table
axis off;
xlim([0 1]);
ylim([0 1]);
set(gca, 'Color', 'w');

% Display the title
text(0.5, 0.95, 'RLE Output of First 8x8 Block', ...
     'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

% Display column headers
text(0.1, 0.8, 'Pair Index:', 'FontSize', 15, 'FontWeight', 'bold');
text(0.5, 0.8, 'RLE Value:', 'FontSize', 15, 'FontWeight', 'bold');
text(0.8, 0.8, 'Count:', 'FontSize', 15, 'FontWeight', 'bold');

% Display each RLE pair (index, value, count) as a row in the table
num_pairs = length(values);
for i = 1:min(num_pairs, 10) 
    y_position = 0.75 - (i-1) * 0.05; % To stack rows vertically
    text(0.1, y_position, sprintf('%d', i), 'FontSize', 10);
    text(0.5, y_position, sprintf('%.1f', values(i)), 'FontSize', 10);
    text(0.8, y_position, sprintf('%d', counts(i)), 'FontSize', 10);
end

%% ==== STEP 7: Image Reconstruction ==== %%

% Initialise array for reconstructed blocks
reconstructed_blocks = zeros(size(blocks)); 

% Dequantize and apply inverse DCT to each block
for k = 1:total_blocks
    dequantized_block = quantized_blocks(:,:,k) .* jpeg_quant_matrix; 
    reconstructed_blocks(:,:,k) = idct2(dequantized_block); 

    % Shift the reconstructed block back by adding 128 to reverse the centering from Step 2
    reconstructed_blocks(:,:,k) = reconstructed_blocks(:,:,k) + 128;
end

% Reassemble the image from blocks
reconstructed_image = zeros(rows, cols);
block_idx = 1;

for i = 1:block_size:rows
    for j = 1:block_size:cols
        reconstructed_image(i:i+block_size-1, j:j+block_size-1) = reconstructed_blocks(:,:,block_idx);
        block_idx = block_idx + 1;
    end
end

% Convert original grayscale image to double for comparison
gray_image_double = double(gray_image);

% Calculate Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)
mse = mean((gray_image_double - reconstructed_image).^2, 'all');
psnr_value = 10 * log10((255^2) / mse);

% Clip values to [0, 255] and convert to uint8 for display
reconstructed_image_uint8 = uint8(min(max(reconstructed_image, 0), 255));

% Compute the difference image (absolute error) for visualization
difference_image = abs(gray_image_double - reconstructed_image);

% Scale for visibility
difference_image_uint8 = uint8(difference_image * (255 / max(difference_image(:)))); 

% Show original, reconstructed, and difference images side by side
figure('Name', 'Step 7: Image Reconstruction and Quality Analysis');

% Subplot 1: Original grayscale image
subplot(1, 3, 1);
imshow(gray_image);
title('Original Grayscale Image');

% Subplot 2: Reconstructed image
subplot(1, 3, 2);
imshow(reconstructed_image_uint8);
title('Reconstructed Image (Standard Quantization)');

% Subplot 3: Difference image
subplot(1, 3, 3);
imshow(difference_image_uint8);
title('Difference Image (Scaled Absolute Error)');

% Add MSE and PSNR as text on the figure
sgtitle(sprintf('Step 7: Image Reconstruction\nMSE: %.2f, PSNR: %.2f dB', mse, psnr_value));

%% ==== STEP 7.5 Analysing Image Reconstruction at Regions-of Interest (ROIs) ==== %%

% To be able to inspect the effects of JPEG compression, I have focussed in
% on specifc 75x75 Regions-of-Interest (ROIs) on Lenna...

% Define ROI parameters
ROI_dim = 75; 

ROI_coords = [
    240, 250;  % ROI1
    200, 200;  % ROI2
    280, 280   % ROI3
];

num_ROIs = size(ROI_coords, 1);

% Initialise arrays to store ROI slices
ROI_images = cell(num_ROIs, 1); % Original images
ROI_reconstructed = cell(num_ROIs, 1); % Reconstructed images
ROI_difference = cell(num_ROIs, 1); % Difference images

% Extract ROIs
for i = 1:num_ROIs
    row_start = ROI_coords(i, 1);
    col_start = ROI_coords(i, 2);
    row_end = row_start + (ROI_dim - 1);
    col_end = col_start + (ROI_dim - 1);
    
    % Extract slices for each image type
    ROI_images{i} = gray_image(row_start:row_end, col_start:col_end);
    ROI_reconstructed{i} = reconstructed_image_uint8(row_start:row_end, col_start:col_end);
    ROI_difference{i} = difference_image_uint8(row_start:row_end, col_start:col_end);
end

% Display ROIs in a 3x3 grid
figure('Name', 'Step 7.5: ROI Analysis');

% Row 1: ROI1 (Original, Reconstructed, Difference)
subplot(3, 3, 1);
imshow(ROI_images{1});
title('ROI1: Original');

subplot(3, 3, 2);
imshow(ROI_reconstructed{1});
title('ROI1: Reconstructed');

subplot(3, 3, 3);
imshow(ROI_difference{1});
title('ROI1: Difference');

% Row 2: ROI2 (Original, Reconstructed, Difference)
subplot(3, 3, 4);
imshow(ROI_images{2});
title('ROI2: Original');

subplot(3, 3, 5);
imshow(ROI_reconstructed{2});
title('ROI2: Reconstructed');

subplot(3, 3, 6);
imshow(ROI_difference{2});
title('ROI2: Difference');

% Row 3: ROI3 (Original, Reconstructed, Difference)
subplot(3, 3, 7);
imshow(ROI_images{3});
title('ROI3: Original');

subplot(3, 3, 8);
imshow(ROI_reconstructed{3});
title('ROI3: Reconstructed');

subplot(3, 3, 9);
imshow(ROI_difference{3});
title('ROI3: Difference');

% Adjust layout for better spacing
sgtitle('Step 7.5: ROI Analysis (Original, Reconstructed, Difference)');
