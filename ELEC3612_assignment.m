% ELEC3612 Digital Media Engineering
% Assignment: Implementation of JPEG Encoding in MATLAB
% Harry Carless

% Clear workspace and close all figures
clear;
close all;

%% ==== STEP 1: READ AND DISPLAY IMAGE ==== %%
% -------------------------------------------------------------------------
% Load image and convert from RGB to YCbCr colour space
% -------------------------------------------------------------------------
image = imread("peppers.png");
ycbcr_image = rgb2ycbcr(image);

% -------------------------------------------------------------------------
% Extract the Y, Cb and Cr channels
% -------------------------------------------------------------------------
Y = ycbcr_image(:,:,1);
Cb = ycbcr_image(:,:,2);
Cr = ycbcr_image(:,:,3);

% -------------------------------------------------------------------------
% Create pseudo-colour representations for the chrominance channels so that
% they can be displayed with representative colours in a figure.
% -------------------------------------------------------------------------
neutral = uint8(128 * ones(size(Y))); % Constant luminance and neutral for unused channels

% For Cb: combine constant Y and neutral Cr
Cb_img = ycbcr2rgb(cat(3, neutral, Cb, neutral));

% For Cr: combine constant Y and neutral Cb
Cr_img = ycbcr2rgb(cat(3, neutral, neutral, Cr));

% -------------------------------------------------------------------------
% Display the Original RGB image and each YCbCr channel in a 1x4 grid
% -------------------------------------------------------------------------
figure('Name', 'Step 1: Image Loading');
subplot(1,4,1);
imshow(image);
title('Original RGB');

subplot(1,4,2);
imshow(Y);
title('Luminance (Y)');

subplot(1,4,3);
imshow(Cb_img);
title('Chrominance (Cb)');

subplot(1,4,4);
imshow(Cr_img);
title('Chrominance (Cr)');

%% ==== STEP 2: Divide Channels into 8x8 Blocks ==== %%
block_size = 8;

% ----------------------------
% Process Luminance Channel (Y)
% ----------------------------
[rowsY, colsY] = size(Y);
num_blocks_Y = (rowsY / block_size) * (colsY / block_size);
blocks_cell_Y = cell(1, num_blocks_Y);
block_idx = 1;

% Divide Y into 8x8 blocks and centre each block by subtracting 128
for i = 1:block_size:rowsY
    for j = 1:block_size:colsY
        block = double(Y(i:i+block_size-1, j:j+block_size-1)) - 128;
        blocks_cell_Y{block_idx} = block;
        block_idx = block_idx + 1;
    end
end
blocks_Y = cat(3, blocks_cell_Y{:});

% ----------------------------
% Process Chrominance Channel (Cb)
% ----------------------------
% Apply 4:2:0 chroma subsampling: reduce resolution by half
Cb_sub = imresize(Cb, 0.5, 'bilinear');
[rowsC, colsC] = size(Cb_sub);
num_blocks_C = (rowsC / block_size) * (colsC / block_size);
blocks_cell_Cb = cell(1, num_blocks_C);
block_idx = 1;

% Divide subsampled Cb into 8x8 blocks and centre by subtracting 128
for i = 1:block_size:rowsC
    for j = 1:block_size:colsC
        block = double(Cb_sub(i:i+block_size-1, j:j+block_size-1)) - 128;
        blocks_cell_Cb{block_idx} = block;
        block_idx = block_idx + 1;
    end
end
blocks_Cb = cat(3, blocks_cell_Cb{:});

% ----------------------------
% Process Chrominance Channel (Cr)
% ----------------------------
% Apply 4:2:0 chroma subsampling: reduce resolution by half
Cr_sub = imresize(Cr, 0.5, 'bilinear');
[rowsCr, colsCr] = size(Cr_sub);
num_blocks_Cr = (rowsCr / block_size) * (colsCr / block_size);
blocks_cell_Cr = cell(1, num_blocks_Cr);
block_idx = 1;

% Divide subsampled Cr into 8x8 blocks and centre by subtracting 128
for i = 1:block_size:rowsCr
    for j = 1:block_size:colsCr
        block = double(Cr_sub(i:i+block_size-1, j:j+block_size-1)) - 128;
        blocks_cell_Cr{block_idx} = block;
        block_idx = block_idx + 1;
    end
end
blocks_Cr = cat(3, blocks_cell_Cr{:});

% -------------------------------------------------------------------------
% Display Luminance (Y) and Chrominance (Cb_sub and Cr_sub) with 8x8 grid overlay
% -------------------------------------------------------------------------
figure('Name', 'Step 2: 8x8 Blocks for Y, Cb, and Cr');

% Plot Y channel with grid overlay
subplot(1,3,1);
imshow(Y);
hold on;
for x = block_size:block_size:colsY
    line([x, x], [1, rowsY], 'Color', 'r', 'LineWidth', 0.5);
end
for y = block_size:block_size:rowsY
    line([1, colsY], [y, y], 'Color', 'r', 'LineWidth', 0.5);
end
title('Luminance (Y) with 8x8 Grid');
xlabel('Columns'); ylabel('Rows');
hold off;

% Plot Cb channel with grid overlay
subplot(1,3,2);
imshow(Cb_sub);
hold on;
for x = block_size:block_size:colsC
    line([x, x], [1, rowsC], 'Color', 'g', 'LineWidth', 0.5);
end
for y = block_size:block_size:rowsC
    line([1, colsC], [y, y], 'Color', 'g', 'LineWidth', 0.5);
end
title('Chrominance (Cb) with 8x8 Grid (Subsampled)');
xlabel('Columns'); ylabel('Rows');
hold off;

% Plot Cr channel with grid overlay
subplot(1,3,3);
imshow(Cr_sub);
hold on;
for x = block_size:block_size:colsCr
    line([x, x], [1, rowsCr], 'Color', 'b', 'LineWidth', 0.5);
end
for y = block_size:block_size:rowsCr
    line([1, colsCr], [y, y], 'Color', 'b', 'LineWidth', 0.5);
end
title('Chrominance (Cr) with 8x8 Grid (Subsampled)');
xlabel('Columns'); ylabel('Rows');
hold off;

%% ==== STEP 3: Apply Discrete Cosine Transform (DCT) ==== %%
% Apply the 2D-DCT to each 8x8 block for each channel and compute a log-scaled 
% version of the DCT coefficients for visualisation. A common colour scale is 
% determined for consistent comparison.

selected_block = 250;

% ----------------------
% Process Luminance Channel (Y)
% ----------------------
num_blocks_Y = size(blocks_Y, 3);
dct_blocks_Y = zeros(size(blocks_Y));
for k = 1:num_blocks_Y
    dct_blocks_Y(:,:,k) = dct2(blocks_Y(:,:,k));
end
selected_dct_block_Y = dct_blocks_Y(:,:,selected_block);
dct_display_Y = abs(selected_dct_block_Y) + 1e-5;
log_dct_Y = log10(dct_display_Y);

% -------------------------
% Process Chrominance Channel (Cb)
% -------------------------
num_blocks_Cb = size(blocks_Cb, 3);
dct_blocks_Cb = zeros(size(blocks_Cb));
for k = 1:num_blocks_Cb
    dct_blocks_Cb(:,:,k) = dct2(blocks_Cb(:,:,k));
end
selected_dct_block_Cb = dct_blocks_Cb(:,:,selected_block);
dct_display_Cb = abs(selected_dct_block_Cb) + 1e-5;
log_dct_Cb = log10(dct_display_Cb);

% -------------------------
% Process Chrominance Channel (Cr)
% -------------------------
num_blocks_Cr = size(blocks_Cr, 3);
dct_blocks_Cr = zeros(size(blocks_Cr));
for k = 1:num_blocks_Cr
    dct_blocks_Cr(:,:,k) = dct2(blocks_Cr(:,:,k));
end
selected_dct_block_Cr = dct_blocks_Cr(:,:,selected_block);
dct_display_Cr = abs(selected_dct_block_Cr) + 1e-5;
log_dct_Cr = log10(dct_display_Cr);

% -------------------------------------------------------------------------
% Determine a common colour scale from all channels
% -------------------------------------------------------------------------
global_min = min([min(log_dct_Y(:)), min(log_dct_Cb(:)), min(log_dct_Cr(:))]);
global_max = max([max(log_dct_Y(:)), max(log_dct_Cb(:)), max(log_dct_Cr(:))]);

% -------------------------------------------------------------------------
% Visualise the DCT Coefficients using the helper function createHeatmap
% -------------------------------------------------------------------------
figure('Name', 'Step 3: DCT Coefficients for Y, Cb, and Cr');

subplot(1,3,1);
createHeatmap(log_dct_Y, selected_dct_block_Y, 'DCT Coefficients (Y)', ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max]);

subplot(1,3,2);
createHeatmap(log_dct_Cb, selected_dct_block_Cb, 'DCT Coefficients (Cb)', ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max]);

subplot(1,3,3);
createHeatmap(log_dct_Cr, selected_dct_block_Cr, 'DCT Coefficients (Cr)', ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max]);

%% ==== STEP 4: Quantisation ==== %%
% Quantise the DCT coefficients for the luminance and chrominance channels 
% using standard JPEG quantisation matrices.

% Define Quantisation Matrices
jpeg_quant_matrix_Y = [ ...
    16 11 10 16 24 40 51 61; ...
    12 12 14 19 26 58 60 55; ...
    14 13 16 24 40 57 69 56; ...
    14 17 22 29 51 87 80 62; ...
    18 22 37 56 68 109 103 77; ...
    24 35 55 64 81 104 113 92; ...
    49 64 78 87 103 121 120 101; ...
    72 92 95 98 112 100 103 99];

jpeg_quant_matrix_chroma = [ ...
    17 18 24 47 99 99 99 99; ...
    18 21 26 66 99 99 99 99; ...
    24 26 56 99 99 99 99 99; ...
    47 66 99 99 99 99 99 99; ...
    99 99 99 99 99 99 99 99; ...
    99 99 99 99 99 99 99 99; ...
    99 99 99 99 99 99 99 99; ...
    99 99 99 99 99 99 99 99];

% -------------------------------
% Quantisation for Luminance (Y)
% -------------------------------
quantized_blocks_Y = zeros(size(dct_blocks_Y));
for k = 1:num_blocks_Y
    quantized_blocks_Y(:,:,k) = round(dct_blocks_Y(:,:,k) ./ jpeg_quant_matrix_Y);
end
selected_quantized_block_Y = quantized_blocks_Y(:,:,selected_block);
quant_display_Y = abs(selected_quantized_block_Y) + 1e-5;
log_quant_Y = log10(quant_display_Y);

% -------------------------------
% Quantisation for Chrominance (Cb)
% -------------------------------
quantized_blocks_Cb = zeros(size(dct_blocks_Cb));
for k = 1:num_blocks_Cb
    quantized_blocks_Cb(:,:,k) = round(dct_blocks_Cb(:,:,k) ./ jpeg_quant_matrix_chroma);
end
selected_quantized_block_Cb = quantized_blocks_Cb(:,:,selected_block);
quant_display_Cb = abs(selected_quantized_block_Cb) + 1e-5;
log_quant_Cb = log10(quant_display_Cb);

% -------------------------------
% Quantisation for Chrominance (Cr)
% -------------------------------
quantized_blocks_Cr = zeros(size(dct_blocks_Cr));
for k = 1:num_blocks_Cr
    quantized_blocks_Cr(:,:,k) = round(dct_blocks_Cr(:,:,k) ./ jpeg_quant_matrix_chroma);
end
selected_quantized_block_Cr = quantized_blocks_Cr(:,:,selected_block);
quant_display_Cr = abs(selected_quantized_block_Cr) + 1e-5;
log_quant_Cr = log10(quant_display_Cr);

% -------------------------------------------------------------------------
% Determine a common colour scale for quantised coefficients
% -------------------------------------------------------------------------
global_min = min([min(log_quant_Y(:)), min(log_quant_Cb(:)), min(log_quant_Cr(:))]);
global_max = max([max(log_quant_Y(:)), max(log_quant_Cb(:)), max(log_quant_Cr(:))]);

% -------------------------------------------------------------------------
% Visualise the Quantised DCT Coefficients using createHeatmap
% -------------------------------------------------------------------------
figure('Name', 'Step 4: Quantised DCT Coefficients');

subplot(1,3,1);
createHeatmap(log_quant_Y, selected_quantized_block_Y, 'Quantised DCT Coefficients (Y)', ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max]);

subplot(1,3,2);
createHeatmap(log_quant_Cb, selected_quantized_block_Cb, 'Quantised DCT Coefficients (Cb)', ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max]);

subplot(1,3,3);
createHeatmap(log_quant_Cr, selected_quantized_block_Cr, 'Quantised DCT Coefficients (Cr)', ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max]);

%% ==== STEP 5: ZIG-ZAG Scanning ==== %%
% Generate the zig-zag order and apply it to each quantised block.

selected_block = 250;
zigzag_order = generate_zigzag_order(block_size);

% Process Luminance Channel (Y)
scanned_vectors_Y = zeros(num_blocks_Y, block_size * block_size);
for k = 1:num_blocks_Y
    block = quantized_blocks_Y(:,:,k);
    scanned_vectors_Y(k,:) = block(zigzag_order);
end

% Reconstruct the selected block from the scanned vector
display_block_Y = zeros(block_size);
selected_vector_Y = scanned_vectors_Y(selected_block,:);
for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    display_block_Y(row, col) = selected_vector_Y(idx);
end
log_display_Y = log10(abs(display_block_Y) + 1e-5);

% Process Chrominance Channel (Cb)
scanned_vectors_Cb = zeros(num_blocks_Cb, block_size * block_size);
for k = 1:num_blocks_Cb
    block = quantized_blocks_Cb(:,:,k);
    scanned_vectors_Cb(k,:) = block(zigzag_order);
end
display_block_Cb = zeros(block_size);
selected_vector_Cb = scanned_vectors_Cb(selected_block,:);
for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    display_block_Cb(row, col) = selected_vector_Cb(idx);
end
log_display_Cb = log10(abs(display_block_Cb) + 1e-5);

% Process Chrominance Channel (Cr)
scanned_vectors_Cr = zeros(num_blocks_Cr, block_size * block_size);
for k = 1:num_blocks_Cr
    block = quantized_blocks_Cr(:,:,k);
    scanned_vectors_Cr(k,:) = block(zigzag_order);
end
display_block_Cr = zeros(block_size);
selected_vector_Cr = scanned_vectors_Cr(selected_block,:);
for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    display_block_Cr(row, col) = selected_vector_Cr(idx);
end
log_display_Cr = log10(abs(display_block_Cr) + 1e-5);

% For the overlay in the zig-zag heatmaps we wish to show the order index.
overlayIndices = zeros(block_size);
for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    overlayIndices(row, col) = idx;
end

% -------------------------------------------------------------------------
% Visualise the Zig-Zag Scanned Blocks using createHeatmap
% -------------------------------------------------------------------------
figure('Name', 'Step 5: Zig-Zag Scanned Coefficients (Log Scale)');

subplot(1,3,1);
createHeatmap(log_display_Y, overlayIndices, sprintf('Zig-Zag (Y), Block #%d', selected_block), ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max], '%d');

subplot(1,3,2);
createHeatmap(log_display_Cb, overlayIndices, sprintf('Zig-Zag (Cb), Block #%d', selected_block), ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max], '%d');

subplot(1,3,3);
createHeatmap(log_display_Cr, overlayIndices, sprintf('Zig-Zag (Cr), Block #%d', selected_block), ...
    'Horizontal Frequency', 'Vertical Frequency', [global_min, global_max], '%d');

%% ==== RUN-LENGTH ENCODING FUNCTION ==== %%
function [values, counts] = rle_encode(vector)
    % Perform run-length encoding on the input vector.
    % Outputs:
    %   values - the unique values in the run
    %   counts - the count for each value in the run
    
    values = vector(1);
    counts = 1;
    
    for i = 2:length(vector)
        if vector(i) == vector(i-1)
            counts(end) = counts(end) + 1;
        else        
            values = [values, vector(i)];
            counts = [counts, 1];
        end
    end
end

%% ==== STEP 6: Run-Length Encoding (RLE) ==== %%
[values_Y, counts_Y] = rle_encode(scanned_vectors_Y(selected_block,:));
[values_Cb, counts_Cb] = rle_encode(scanned_vectors_Cb(selected_block,:));
[values_Cr, counts_Cr] = rle_encode(scanned_vectors_Cr(selected_block,:));

% Display the RLE output for the selected block of each channel
figure('Name', 'Step 6: RLE Output of Selected Block', 'Position', [100, 100, 800, 600]);

% ------------------------
% Luminance (Y) RLE Output
% ------------------------
subplot(3,1,1);
axis off;
text(0.5, 0.95, sprintf('RLE Output for Luminance (Y) - Block #%d', selected_block), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.8, 'Pair Index:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.5, 0.8, 'RLE Value:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.8, 0.8, 'Count:', 'FontSize', 12, 'FontWeight', 'bold');
num_pairs = length(values_Y);
for i = 1:num_pairs
    y_position = 0.75 - (i-1)*0.05;
    text(0.1, y_position, sprintf('%d', i), 'FontSize', 10);
    text(0.5, y_position, sprintf('%.1f', values_Y(i)), 'FontSize', 10);
    text(0.8, y_position, sprintf('%d', counts_Y(i)), 'FontSize', 10);
end

% ----------------------------
% Chrominance (Cb) RLE Output
% ----------------------------
subplot(3,1,2);
axis off;
text(0.5, 0.95, sprintf('RLE Output for Chrominance (Cb) - Block #%d', selected_block), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.8, 'Pair Index:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.5, 0.8, 'RLE Value:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.8, 0.8, 'Count:', 'FontSize', 12, 'FontWeight', 'bold');
num_pairs = length(values_Cb);
for i = 1:num_pairs
    y_position = 0.75 - (i-1)*0.05;
    text(0.1, y_position, sprintf('%d', i), 'FontSize', 10);
    text(0.5, y_position, sprintf('%.1f', values_Cb(i)), 'FontSize', 10);
    text(0.8, y_position, sprintf('%d', counts_Cb(i)), 'FontSize', 10);
end

% ---------------------------
% Chrominance (Cr) RLE Output
% ---------------------------
subplot(3,1,3);
axis off;
text(0.5, 0.95, sprintf('RLE Output for Chrominance (Cr) - Block #%d', selected_block), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.8, 'Pair Index:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.5, 0.8, 'RLE Value:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.8, 0.8, 'Count:', 'FontSize', 12, 'FontWeight', 'bold');
num_pairs = length(values_Cr);
for i = 1:num_pairs
    y_position = 0.75 - (i-1)*0.05;
    text(0.1, y_position, sprintf('%d', i), 'FontSize', 10);
    text(0.5, y_position, sprintf('%.1f', values_Cr(i)), 'FontSize', 10);
    text(0.8, y_position, sprintf('%d', counts_Cr(i)), 'FontSize', 10);
end

%% ==== STEP 7: Image Reconstruction ==== %%
% Reconstruct the image from the quantised and DCT-processed blocks for each
% YCbCr channel, combine them and convert back to RGB. Quality metrics (MSE
% and PSNR) are computed for analysis.

% ---------------------------------
% Reconstruct Luminance (Y) Channel
% ---------------------------------
reconstructed_blocks_Y = zeros(size(blocks_Y));
for k = 1:num_blocks_Y
    dequantised_block = quantized_blocks_Y(:,:,k) .* jpeg_quant_matrix_Y;
    reconstructed_block = idct2(dequantised_block);
    reconstructed_blocks_Y(:,:,k) = reconstructed_block + 128;
end

% Reassemble full Y channel from blocks
reconstructed_Y = zeros(rowsY, colsY);
block_idx = 1;
for i = 1:block_size:rowsY
    for j = 1:block_size:colsY 
        reconstructed_Y(i:i+block_size-1, j:j+block_size-1) = reconstructed_blocks_Y(:,:,block_idx);
        block_idx = block_idx + 1;
    end
end

% ------------------------------------
% Reconstruct Chrominance (Cb) Channel
% ------------------------------------
reconstructed_blocks_Cb = zeros(size(blocks_Cb));
for k = 1:num_blocks_C
    dequantised_block = quantized_blocks_Cb(:,:,k) .* jpeg_quant_matrix_chroma;
    reconstructed_block = idct2(dequantised_block);
    reconstructed_blocks_Cb(:,:,k) = reconstructed_block + 128;
end
reconstructed_Cb = zeros(rowsC, colsC);
block_idx = 1;
for i = 1:block_size:rowsC
    for j = 1:block_size:colsC
        reconstructed_Cb(i:i+block_size-1, j:j+block_size-1) = reconstructed_blocks_Cb(:,:,block_idx);
        block_idx = block_idx + 1;
    end
end

% ------------------------------------
% Reconstruct Chrominance (Cr) Channel
% ------------------------------------
reconstructed_blocks_Cr = zeros(size(blocks_Cr));
for k = 1:num_blocks_Cr
    dequantised_block = quantized_blocks_Cr(:,:,k) .* jpeg_quant_matrix_chroma;
    reconstructed_block = idct2(dequantised_block);
    reconstructed_blocks_Cr(:,:,k) = reconstructed_block + 128;
end
reconstructed_Cr = zeros(rowsCr, colsCr);
block_idx = 1;
for i = 1:block_size:rowsCr
    for j = 1:block_size:colsCr
        reconstructed_Cr(i:i+block_size-1, j:j+block_size-1) = reconstructed_blocks_Cr(:,:,block_idx);
        block_idx = block_idx + 1;
    end
end

% ---------------------------------
% Upsample the Chrominance Channels
% ---------------------------------
reconstructed_Cb_up = imresize(reconstructed_Cb, [rowsY, colsY], 'bilinear');
reconstructed_Cr_up = imresize(reconstructed_Cr, [rowsY, colsY], 'bilinear');

% -----------------------------------
% Combine Channels and Convert to RGB
% -----------------------------------
reconstructed_ycbcr = cat(3, uint8(reconstructed_Y), uint8(reconstructed_Cb_up), uint8(reconstructed_Cr_up));
reconstructed_rgb = ycbcr2rgb(reconstructed_ycbcr);

% -------------------------------
% Calculate Quality Metrics
% -------------------------------
original_rgb_double = double(image);
reconstructed_rgb_double = double(reconstructed_rgb);
mse = mean((original_rgb_double - reconstructed_rgb_double).^2, 'all');
psnr_value = 10 * log10((255^2) / mse);
difference_image = abs(original_rgb_double - reconstructed_rgb_double);
difference_image_uint8 = uint8(difference_image * (255 / max(difference_image(:))));

% -------------------------------------------------------------
% Visualise the Original, Reconstructed, and Difference Images
% -------------------------------------------------------------
figure('Name', 'Step 7: Image Reconstruction and Quality Analysis');

subplot(1, 3, 1);
imshow(image);
title('Original RGB Image');

subplot(1, 3, 2);
imshow(reconstructed_rgb);
title('Reconstructed RGB Image');

subplot(1, 3, 3);
imshow(difference_image_uint8);
title('Difference Image (Scaled Absolute Error)');

sgtitle(sprintf('Image Reconstruction\nMSE: %.2f, PSNR: %.2f dB', mse, psnr_value));

%% ---- Stage 8: Custom Quantisation Matrix and Reconstruction ---- 
% Compare processing using the standard and a custom (more aggressive) 
% quantisation matrix for the Y channel.
% The custom quantisation matrix is applied to the luminance channel.
% In addition to comparing the quantised DCT coefficients, the image is 
% reconstructed using both the standard and custom quantisation for the Y channel.
% Quality metrics (MSE and PSNR) are computed for both the full RGB images 
% and the isolated Y channels.

% Define a custom quantisation matrix (scaled 2.5x standard)
custom_quantisation_matrix = jpeg_quant_matrix_Y * 2.5;

% -------------------------------
% Quantisation for Luminance (Y) with custom matrix
% -------------------------------
quantized_blocks_Y_custom = zeros(size(dct_blocks_Y)); 
for k = 1:num_blocks_Y
    quantized_blocks_Y_custom(:,:,k) = round(dct_blocks_Y(:,:,k) ./ custom_quantisation_matrix);
end
selected_quantized_block_Y_custom = quantized_blocks_Y_custom(:,:,selected_block);
quant_display_Y_custom = abs(selected_quantized_block_Y_custom) + 1e-5;
log_quant_Y_custom = log10(quant_display_Y_custom);

% Determine a common colour scale for the heatmap comparison
y_global_min = min([min(log_quant_Y(:)), min(log_quant_Y_custom(:))]);
y_global_max = max([max(log_quant_Y(:)), max(log_quant_Y_custom(:))]);

% Visualise the comparison of quantised DCT coefficients for the Y channel
figure('Name', 'Stage 8: Quantisation Comparison (Y Channel)');
subplot(1,2,1);
createHeatmap(log_quant_Y, selected_quantized_block_Y, 'Quantised DCT Coefficients (Y Standard)', ...
    'Horizontal Frequency', 'Vertical Frequency', [y_global_min, y_global_max]);
subplot(1,2,2);
createHeatmap(log_quant_Y_custom, selected_quantized_block_Y_custom, 'Quantised DCT Coefficients (Y Custom)', ...
    'Horizontal Frequency', 'Vertical Frequency', [y_global_min, y_global_max]);

% ----- Reconstruction using custom quantisation for Y channel -----
% Reconstruct the Y channel using the custom quantisation matrix
reconstructed_blocks_Y_custom = zeros(size(blocks_Y));
for k = 1:num_blocks_Y
    dequantised_block_custom = quantized_blocks_Y_custom(:,:,k) .* custom_quantisation_matrix;
    reconstructed_block_custom = idct2(dequantised_block_custom);
    reconstructed_blocks_Y_custom(:,:,k) = reconstructed_block_custom + 128;
end

% Reassemble the full Y channel for custom quantisation
reconstructed_Y_custom = zeros(rowsY, colsY);
block_idx = 1;
for i = 1:block_size:rowsY
    for j = 1:block_size:colsY 
        reconstructed_Y_custom(i:i+block_size-1, j:j+block_size-1) = reconstructed_blocks_Y_custom(:,:,block_idx);
        block_idx = block_idx + 1;
    end
end

% Combine the custom-reconstructed Y channel with the standard reconstructed Cb and Cr channels
reconstructed_ycbcr_custom = cat(3, uint8(reconstructed_Y_custom), uint8(reconstructed_Cb_up), uint8(reconstructed_Cr_up));
reconstructed_rgb_custom = ycbcr2rgb(reconstructed_ycbcr_custom);

% ----- Calculate quality metrics for both reconstructions -----
% Standard reconstruction (from Stage 7) is assumed stored in 'reconstructed_rgb'
% For the full RGB images:
original_rgb_double = double(image);
reconstructed_rgb_double = double(reconstructed_rgb);
mse = mean((original_rgb_double - reconstructed_rgb_double).^2, 'all');
psnr_value = 10 * log10((255^2) / mse);

reconstructed_rgb_custom_double = double(reconstructed_rgb_custom);
mse_custom = mean((original_rgb_double - reconstructed_rgb_custom_double).^2, 'all');
psnr_custom = 10 * log10((255^2) / mse_custom);

% For the isolated Y channels, compare with the original Y channel
original_Y_double = double(Y);
reconstructed_Y_double = double(reconstructed_Y);
mse_Y_standard = mean((original_Y_double - reconstructed_Y_double).^2, 'all');
psnr_Y_standard = 10 * log10((255^2) / mse_Y_standard);

reconstructed_Y_custom_double = double(reconstructed_Y_custom);
mse_Y_custom = mean((original_Y_double - reconstructed_Y_custom_double).^2, 'all');
psnr_Y_custom = 10 * log10((255^2) / mse_Y_custom);

% ----- Display side-by-side comparison of reconstructed images and their Y channels -----
figure('Name', 'Stage 8: Reconstructed Images & Y Channels Comparison', ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
t = tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Standard RGB reconstruction
nexttile;
imshow(reconstructed_rgb);
title(sprintf('Standard Quantisation\nRGB: MSE: %.2f, PSNR: %.2f dB', mse, psnr_value));

% Custom RGB reconstruction
nexttile;
imshow(reconstructed_rgb_custom);
title(sprintf('Custom Quantisation\nRGB: MSE: %.2f, PSNR: %.2f dB', mse_custom, psnr_custom));

% Standard isolated Y channel
nexttile;
imshow(uint8(reconstructed_Y));
title(sprintf('Standard Quantisation - Isolated Y\nMSE: %.2f, PSNR: %.2f dB', mse_Y_standard, psnr_Y_standard));

% Custom isolated Y channel
nexttile;
imshow(uint8(reconstructed_Y_custom));
title(sprintf('Custom Quantisation - Isolated Y\nMSE: %.2f, PSNR: %.2f dB', mse_Y_custom, psnr_Y_custom));

%% ===== Helper Function: createHeatmap =====
function createHeatmap(heatmapData, overlayData, titleStr, xLabelStr, yLabelStr, climVals, overlayFormat)
    % createHeatmap plots a heatmap with an overlaid text grid.
    %
    % Inputs:
    %   heatmapData   - matrix data to be displayed (e.g. log-scaled coefficients)
    %   overlayData   - matrix of values to overlay as text; should be of the same size
    %                   as heatmapData
    %   titleStr      - title of the plot
    %   xLabelStr     - label for the x-axis
    %   yLabelStr     - label for the y-axis
    %   climVals      - two-element vector specifying the [min max] colour limits
    %   overlayFormat - (optional) format string for text overlay (default is '%.1f')
    
    if nargin < 7
        overlayFormat = '%.1f';
    end
    
    imagesc(heatmapData);
    title(titleStr);
    xlabel(xLabelStr);
    ylabel(yLabelStr);
    
    nrows = size(heatmapData, 1);
    ncols = size(heatmapData, 2);
    set(gca, 'XTick', 0.5:1:(ncols+0.5), 'YTick', 0.5:1:(nrows+0.5));
    grid on;
    axis square;
    clim(climVals);
    hold on;
    
    for r = 1:nrows
        for c = 1:ncols
            text(c, r, sprintf(overlayFormat, overlayData(r,c)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'Color', 'black', 'FontSize', 8);
        end
    end
    hold off;
end

%% ===== Function: generate_zigzag_order =====
function zigzag_order = generate_zigzag_order(block_size)
    % generate_zigzag_order generates the zig-zag scanning order for an NxN block.
    %
    % Input:
    %   block_size  - the size of the block (e.g. 8)
    % Output:
    %   zigzag_order - a vector of linear indices in zig-zag order
    
    N = block_size;
    zigzag_order = zeros(1, N*N); 
    idx = 1;  % Current index in the order vector
    row = 1;  % Start at top-left corner
    col = 1;
    going_up = true; % Direction flag: true for up-right, false for down-left
    
    while idx <= N*N
        zigzag_order(idx) = (row-1)*N + col;
        idx = idx + 1;
        if going_up
            if col == N
                row = row + 1; % At right edge, move down
                going_up = false;
            elseif row == 1
                col = col + 1; % At top edge, move right
                going_up = false;
            else
                row = row - 1;
                col = col + 1;
            end
        else
            if row == N
                col = col + 1; % At bottom edge, move right
                going_up = true;
            elseif col == 1
                row = row + 1; % At left edge, move down
                going_up = true;
            else
                row = row + 1;
                col = col - 1;
            end
        end
    end
end
