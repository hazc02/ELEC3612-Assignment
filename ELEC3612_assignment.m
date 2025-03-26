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

% Initialise array for DCT coefficients and apply DCT to each
dct_blocks = zeros(size(blocks)); 
for k = 1:total_blocks
    dct_blocks(:,:,k) = dct2(blocks(:,:,k)); 
end

% Select block to visualisation (first block)
selected_dct_block = dct_blocks(:,:,1);

% Prepare the DCT coefficients for log-scaled display

% Use absolute values and a small offset to handle negatives and zeros
dct_display = abs(selected_dct_block) + 1e-5;
log_dct = log10(dct_display); % Log-scale for better visibility

% Display the log-scaled DCT coefficients as a heatmap
figure('Name', 'Step 3: DCT Coefficients (Log-Scaled)');

imagesc(log_dct);  % Generate heatmap
colormap parula;   % Using 'parula' a clear, uniform colour gradient
colorbar;          % Show colour-to-value mapping

% Enhanced clarity in figure 
title('Log-Scaled DCT Coefficients of Selected 8x8 Block');
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5); % Add grid lines
grid on;

