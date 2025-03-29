% ELEC3612 Digital Media Engineering
% Assignment: Implementation of JPEG Encoding in MATLAB
% Harry Carless

% Clear workspace and close all figures
clear;
close all;

%% ==== STEP 1: READ AND DISPLAY IMAGE ==== %%

% -----------------------------------------
% Load Image and convert from RGB to YCbCr
% -----------------------------------------

image = imread("peppers.png");
ycbcr_image = rgb2ycbcr(image);

% ----------------------------
% Extract the Y, Cb and Cr channels
% ----------------------------

Y = ycbcr_image(:,:,1);
Cb = ycbcr_image(:,:,2);
Cr = ycbcr_image(:,:,3);

% -----------------------------------------------------------------------
% Creating pseudo-colour representations for the chrominance channels.
% to be able to represent them in a figure with the colours they represent
% ------------------------------------------------------------------------

% Use a constant luminance value (128) and neutral values (128) for the channel not being displayed.
neutral = uint8(128 * ones(size(Y)));

% For Cb: combine constant Y and neutral Cr
Cb_img = ycbcr2rgb(cat(3, neutral, Cb, neutral));

% For Cr: combine constant Y and neutral Cb
Cr_img = ycbcr2rgb(cat(3, neutral, neutral, Cr));

% --------------------------------------------------------------
% Display Original RGB and then each YCbCr Channel in a 1x4 grid
% --------------------------------------------------------------

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
% Luminance Channel (Y)
% ----------------------------

% Initalise arrays to store blocks
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

% Stack blocks into a 3D array
blocks_Y = cat(3, blocks_cell_Y{:});

% ----------------------------
% Chrominance Channel (Cb)
% ----------------------------

% Applying 4:2:0 chroma subsampling: reduce resolution by half
Cb_sub = imresize(Cb, 0.5, 'bilinear');

% Initalise arrays to store blocks
[rowsC, colsC] = size(Cb_sub);
num_blocks_C = (rowsC / block_size) * (colsC / block_size);
blocks_cell_Cb = cell(1, num_blocks_C);

% Resetting Block Index for this channel
block_idx = 1;

% Divide subsampled Cb into 8x8 blocks and centre by subtracting 128
for i = 1:block_size:rowsC
    for j = 1:block_size:colsC
        block = double(Cb_sub(i:i+block_size-1, j:j+block_size-1)) - 128;
        blocks_cell_Cb{block_idx} = block;
        block_idx = block_idx + 1;
    end
end

% Stack blocks into a 3D array
blocks_Cb = cat(3, blocks_cell_Cb{:});

% ----------------------------
% Chrominance Channel (Cr)
% ----------------------------

% Apply 4:2:0 chroma subsampling: reduce resolution by half
Cr_sub = imresize(Cr, 0.5, 'bilinear');

% Initalise arrays to store blocks
[rowsCr, colsCr] = size(Cr_sub);
num_blocks_Cr = (rowsCr / block_size) * (colsCr / block_size);
blocks_cell_Cr = cell(1, num_blocks_Cr);

% Resetting Block Index for this channel
block_idx = 1;

% Divide subsampled Cr into 8x8 blocks and centre by subtracting 128
for i = 1:block_size:rowsCr
    for j = 1:block_size:colsCr
        block = double(Cr_sub(i:i+block_size-1, j:j+block_size-1)) - 128;
        blocks_cell_Cr{block_idx} = block;
        block_idx = block_idx + 1;
    end
end

% Stack blocks into a 3D array
blocks_Cr = cat(3, blocks_cell_Cr{:});

% ----------------------------------------------------------------------
% Display Luminance (Y) and Chrominance (Cb_sub and Cr_sub) Side by Side
% ----------------------------------------------------------------------

figure('Name', 'Step 2: 8x8 Blocks for Y, Cb, and Cr');

% Plot Y Channel
subplot(1,3,1);
imshow(Y);
hold on;

% Grid overlay to show 8x8 blocks for Y
for x = block_size:block_size:colsY
    line([x, x], [1, rowsY], 'Color', 'r', 'LineWidth', 0.5);
end

for y = block_size:block_size:rowsY
    line([1, colsY], [y, y], 'Color', 'r', 'LineWidth', 0.5);
end

% Format Cr Figure
title('Luminance (Y) with 8x8 Grid');
xlabel('Columns'); ylabel('Rows');
hold off;

% Plot Cb Channel
subplot(1,3,2);
imshow(Cb_sub);
hold on;

% Grid overlay to show 8x8 blocks for Cb
for x = block_size:block_size:colsC
    line([x, x], [1, rowsC], 'Color', 'g', 'LineWidth', 0.5);
end
for y = block_size:block_size:rowsC
    line([1, colsC], [y, y], 'Color', 'g', 'LineWidth', 0.5);
end

% Format Cr Figure
title('Chrominance (Cb) with 8x8 Grid (Subsampled)');
xlabel('Columns'); ylabel('Rows');
hold off;

% Plot Cr Channel
subplot(1,3,3);
imshow(Cr_sub);
hold on;

% Grid overlay to show 8x8 blocks for Cr
for x = block_size:block_size:colsCr
    line([x, x], [1, rowsCr], 'Color', 'b', 'LineWidth', 0.5);
end
for y = block_size:block_size:rowsCr
    line([1, colsCr], [y, y], 'Color', 'b', 'LineWidth', 0.5);
end

% Format Cr Figure
title('Chrominance (Cr) with 8x8 Grid (Subsampled)');
xlabel('Columns'); ylabel('Rows');
hold off;


%% ==== STEP 3: Apply Discrete Cosine Transform (DCT) ==== %%
% Apply the 2D-DCT to each 8x8 block for each channel (Y, Cb, Cr) and compute 
% a log-scaled version of the DCT coefficients for visualisation. A common 
% colour scale is then determined across all channels for consistent comparison.

selected_block = 250;

% ----------------------
% Luminance Channel (Y)
% ----------------------

% Compute the DCT for each 8x8 block in the Y channel
num_blocks_Y = size(blocks_Y, 3);
dct_blocks_Y = zeros(size(blocks_Y));
for k = 1:num_blocks_Y
    dct_blocks_Y(:,:,k) = dct2(blocks_Y(:,:,k));
end
selected_dct_block_Y = dct_blocks_Y(:,:,selected_block);

% Compute the absolute DCT coefficients and take log-scale (avoid log(0))
dct_display_Y = abs(selected_dct_block_Y) + 1e-5;
log_dct_Y = log10(dct_display_Y);

% -------------------------
% Chrominance Channel (Cb)
% -------------------------

% Compute the DCT for each 8x8 block in the Cb channel
num_blocks_Cb = size(blocks_Cb, 3);
dct_blocks_Cb = zeros(size(blocks_Cb));
for k = 1:num_blocks_Cb
    dct_blocks_Cb(:,:,k) = dct2(blocks_Cb(:,:,k));
end
selected_dct_block_Cb = dct_blocks_Cb(:,:,selected_block);

% Compute the log-scaled DCT coefficients for Cb
dct_display_Cb = abs(selected_dct_block_Cb) + 1e-5;
log_dct_Cb = log10(dct_display_Cb);

% -------------------------
% Chrominance Channel (Cr)
% -------------------------

% Compute the DCT for each 8x8 block in the Cr channel
num_blocks_Cr = size(blocks_Cr, 3);
dct_blocks_Cr = zeros(size(blocks_Cr));
for k = 1:num_blocks_Cr
    dct_blocks_Cr(:,:,k) = dct2(blocks_Cr(:,:,k));
end
selected_dct_block_Cr = dct_blocks_Cr(:,:,selected_block);

% Compute the log-scaled DCT coefficients for Cr
dct_display_Cr = abs(selected_dct_block_Cr) + 1e-5;
log_dct_Cr = log10(dct_display_Cr);

% -----------------------------------------------
% Compute a Common Colour Scale and Plot Figures
% -----------------------------------------------

% Determine global min and max from all three channels for a unified colour scale
global_min = min([min(log_dct_Y(:)), min(log_dct_Cb(:)), min(log_dct_Cr(:))]);
global_max = max([max(log_dct_Y(:)), max(log_dct_Cb(:)), max(log_dct_Cr(:))]);

% Visualise the Heatmaps for Each Channel 
figure('Name', 'Step 3: DCT Coefficients for Y, Cb, and Cr');

% Luminance (Y) heatmap
subplot(1,3,1);
imagesc(log_dct_Y);
title('DCT Coefficients (Y)');
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5); % Set grid ticks
grid on;
axis square;                       % Ensure square aspect ratio
clim([global_min, global_max]);   % Apply common colour scale
hold on;
for row = 1:8
    for col = 1:8
        text(col, row, sprintf('%.1f', selected_dct_block_Y(row, col)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'black', 'FontSize', 8);
    end
end
hold off;

% Chrominance (Cb) heatmap
subplot(1,3,2);
imagesc(log_dct_Cb);
title('DCT Coefficients (Cb)');
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5);
grid on;
axis square;
clim([global_min, global_max]);
hold on;
for row = 1:8
    for col = 1:8
        text(col, row, sprintf('%.1f', selected_dct_block_Cb(row, col)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'black', 'FontSize', 8);
    end
end
hold off;

% Chrominance (Cr) heatmap
subplot(1,3,3);
imagesc(log_dct_Cr);
title('DCT Coefficients (Cr)');
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5);
grid on;
axis square;
clim([global_min, global_max]);
hold on;
for row = 1:8
    for col = 1:8
        text(col, row, sprintf('%.1f', selected_dct_block_Cr(row, col)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Color', 'black', 'FontSize', 8);
    end
end
hold off;

%% ==== STEP 4: Quantisation ==== %%
% This stage applies quantisation to the DCT coefficients for the luminance and 
% chrominance channels using different quantisation matrices, reflecting the 
% differing perceptual sensitivities of the human eye.

% -------------------------------
% Define Quantisation Matrices
% -------------------------------

% Standard JPEG Luminance Quantisation Matrix (for Y)
jpeg_quant_matrix_Y = [ ...
    16 11 10 16 24 40 51 61; ...
    12 12 14 19 26 58 60 55; ...
    14 13 16 24 40 57 69 56; ...
    14 17 22 29 51 87 80 62; ...
    18 22 37 56 68 109 103 77; ...
    24 35 55 64 81 104 113 92; ...
    49 64 78 87 103 121 120 101; ...
    72 92 95 98 112 100 103 99];

% Standard JPEG Chrominance Quantisation Matrix (for Cb and Cr)
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
num_blocks_Y = size(dct_blocks_Y, 3);
quantized_blocks_Y = zeros(size(dct_blocks_Y));

% Quantise each 8x8 block for Y
for k = 1:num_blocks_Y
    quantized_blocks_Y(:,:,k) = round(dct_blocks_Y(:,:,k) ./ jpeg_quant_matrix_Y);
end

selected_quantized_block_Y = quantized_blocks_Y(:,:,selected_block);
quant_display_Y = abs(selected_quantized_block_Y) + 1e-5;
log_quant_Y = log10(quant_display_Y);

% -------------------------------
% Quantisation for Chrominance (Cb)
% -------------------------------
num_blocks_Cb = size(dct_blocks_Cb, 3);
quantized_blocks_Cb = zeros(size(dct_blocks_Cb));

% Quantise each 8x8 block for Cb
for k = 1:num_blocks_Cb
    quantized_blocks_Cb(:,:,k) = round(dct_blocks_Cb(:,:,k) ./ jpeg_quant_matrix_chroma);
end

selected_quantized_block_Cb = quantized_blocks_Cb(:,:,selected_block);
quant_display_Cb = abs(selected_quantized_block_Cb) + 1e-5;
log_quant_Cb = log10(quant_display_Cb);

% -------------------------------
% Quantisation for Chrominance (Cr)
% -------------------------------
num_blocks_Cr = size(dct_blocks_Cr, 3);
quantized_blocks_Cr = zeros(size(dct_blocks_Cr));

% Quantise each 8x8 block for Cr
for k = 1:num_blocks_Cr
    quantized_blocks_Cr(:,:,k) = round(dct_blocks_Cr(:,:,k) ./ jpeg_quant_matrix_chroma);
end

selected_quantized_block_Cr = quantized_blocks_Cr(:,:,selected_block);
quant_display_Cr = abs(selected_quantized_block_Cr) + 1e-5;
log_quant_Cr = log10(quant_display_Cr);

% ------------------------------------------------------
% Determine a common colour scale across all 3 channels
% ------------------------------------------------------
global_min = min([min(log_quant_Y(:)), min(log_quant_Cb(:)), min(log_quant_Cr(:))]);
global_max = max([max(log_quant_Y(:)), max(log_quant_Cb(:)), max(log_quant_Cr(:))]);

% -------------------------------
% Visualisation: Display Heatmaps
% -------------------------------
figure('Name', 'Step 4: Quantised DCT Coefficients');

% Luminance (Y) Heatmap
subplot(1,3,1);
imagesc(log_quant_Y);
title('Quantised DCT Coefficients (Y)');
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5);
grid on;
axis square;
clim([global_min, global_max]);  % Set common colour scale
hold on;

for row = 1:8
    for col = 1:8
        text(col, row, sprintf('%.1f', selected_quantized_block_Y(row, col)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', 'black', 'FontSize', 8);
    end
end
hold off;

% Chrominance (Cb) Heatmap
subplot(1,3,2);
imagesc(log_quant_Cb);
title('Quantised DCT Coefficients (Cb)');
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5);
grid on;
axis square;
clim([global_min, global_max]);  % Set common colour scale
hold on;

for row = 1:8
    for col = 1:8
        text(col, row, sprintf('%.1f', selected_quantized_block_Cb(row, col)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', 'black', 'FontSize', 8);
    end
end
hold off;

% Chrominance (Cr) Heatmap
subplot(1,3,3);
imagesc(log_quant_Cr);
title('Quantised DCT Coefficients (Cr)');
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5);
grid on;
axis square;
clim([global_min, global_max]);  % Set common colour scale
hold on;
for row = 1:8
    for col = 1:8
        text(col, row, sprintf('%.1f', selected_quantized_block_Cr(row, col)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', 'black', 'FontSize', 8);
    end
end
hold off;

%% ==== ZIGZAG ORDER FUNCTION ==== %%
function zigzag_order = generate_zigzag_order(block_size)
    % Generate the zig-zag scanning order for an NxN block.
    % Input: block_size - the size of the block (e.g. 8)
    % Output: zigzag_order - a vector of indices in zig-zag order

    N = block_size;
    zigzag_order = zeros(1, N*N); 
    idx = 1; % Current index in the order vector
    row = 1; % Start at top-left corner
    col = 1;
    going_up = true; % Direction flag: true for up-right, false for down-left
    
    while idx <= N*N
        zigzag_order(idx) = (row-1)*N + col;
        idx = idx + 1;
        if going_up
            if col == N
                row = row + 1; % Hit right edge, move down
                going_up = false;
            elseif row == 1
                col = col + 1; % Hit top edge, move right
                going_up = false;
            else
                row = row - 1;
                col = col + 1;
            end
        else
            if row == N
                col = col + 1; % Hit bottom edge, move right
                going_up = true;
            elseif col == 1
                row = row + 1; % Hit left edge, move down
                going_up = true;
            else
                row = row + 1;
                col = col - 1;
            end
        end
    end
end

%% ==== STEP 5: ZIG-ZAG Scanning ==== %%

selected_block = 250;

% Generate the zig-zag order (same for all channels)
zigzag_order = generate_zigzag_order(block_size);

% --------------------
% Luminance Channel(Y)
% --------------------

% Create an array to store scanned blocks
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

% Compute the log-scale version for display
log_display_Y = log10(abs(display_block_Y) + 1e-5);

% -------------------------
% Chrominance (Cb) Channel
% -------------------------

% Create an array to store scanned blocks
scanned_vectors_Cb = zeros(num_blocks_Cb, block_size * block_size);

for k = 1:num_blocks_Cb
    block = quantized_blocks_Cb(:,:,k);
    scanned_vectors_Cb(k,:) = block(zigzag_order);
end

% Reconstruct the selected block from the scanned vector
display_block_Cb = zeros(block_size);
selected_vector_Cb = scanned_vectors_Cb(selected_block,:);

for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    display_block_Cb(row, col) = selected_vector_Cb(idx);
end

% Compute the log-scale version
log_display_Cb = log10(abs(display_block_Cb) + 1e-5);

% -------------------------
% Chrominance (Cr) Channel
% -------------------------

% Create an array to store scanned blocks
scanned_vectors_Cr = zeros(num_blocks_Cr, block_size * block_size);

for k = 1:num_blocks_Cr
    block = quantized_blocks_Cr(:,:,k);
    scanned_vectors_Cr(k,:) = block(zigzag_order);
end

% Reconstruct the selected block from the scanned vector
display_block_Cr = zeros(block_size);
selected_vector_Cr = scanned_vectors_Cr(selected_block,:);

for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    display_block_Cr(row, col) = selected_vector_Cr(idx);
end

% Compute the log-scale version
log_display_Cr = log10(abs(display_block_Cr) + 1e-5);

% ----------------------------------------------------------------
% Visualise the Log-Scaled Zig-Zag Blocks with Global Colour Scale
% ----------------------------------------------------------------

figure('Name', 'Step 5: Zig-Zag Scanned Coefficients (Log Scale)');

% Luminance (Y) heatmap
subplot(1,3,1);
imagesc(log_display_Y);
title(sprintf('Zig-Zag (Y), Block #%d', selected_block));
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5);
grid on; axis square;
clim([global_min, global_max]);  % Apply common log-scale
hold on;
for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    text(col, row, sprintf('%d', idx), 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', 'Color', 'black', 'FontSize', 8);
end
hold off;

% Chrominance (Cb) heatmap
subplot(1,3,2);
imagesc(log_display_Cb);
title(sprintf('Zig-Zag (Cb), Block #%d', selected_block));
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5);
grid on; axis square;
clim([global_min, global_max]);
hold on;
for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    text(col, row, sprintf('%d', idx), 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', 'Color', 'black', 'FontSize', 8);
end
hold off;

% Chrominance (Cr) heatmap
subplot(1,3,3);
imagesc(log_display_Cr);
title(sprintf('Zig-Zag (Cr), Block #%d', selected_block));
xlabel('Horizontal Frequency');
ylabel('Vertical Frequency');
set(gca, 'XTick', 0.5:1:8.5, 'YTick', 0.5:1:8.5);
grid on; axis square;
clim([global_min, global_max]);
hold on;
for idx = 1:length(zigzag_order)
    [row, col] = ind2sub([block_size, block_size], zigzag_order(idx));
    text(col, row, sprintf('%d', idx), 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', 'Color', 'black', 'FontSize', 8);
end
hold off;

%% ==== RUN-LENGTH ENCODING FUNCTION ==== %%
function [values, counts] = rle_encode(vector)

    % Perform run-length encoding on the input vector.

    %   Outputs:
    %   values - the unique values in the run
    %   counts - the count for each value in the run
    
    % Initalise values and count
    values = vector(1);
    counts = 1;
    
   
    for i = 2:length(vector)

        % If the values is the same as previous, incriment count
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

% Display the RLE output for the 'selected_block' of each channel
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
% Reconstruct the image from the quantised and DCT-processed blocks for each YCbCr channel.
% The channels are then combined and converted back to RGB for visual comparison.
% Quality metrics (MSE and PSNR) are calculated to assess reconstruction quality.

% ---------------------------------
% Reconstruct Luminance (Y) Channel
% ---------------------------------

reconstructed_blocks_Y = zeros(size(blocks_Y));

for k = 1:num_blocks_Y

    % Dequantise using the luminance quantisation matrix and perform inverse DCT
    dequantised_block = quantized_blocks_Y(:,:,k) .* jpeg_quant_matrix_Y;
    reconstructed_block = idct2(dequantised_block);

    % Revert the centring by adding 128
    reconstructed_blocks_Y(:,:,k) = reconstructed_block + 128;
end

% Reassemble the full Y channel from 8x8 blocks
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
    % Dequantise using the chrominance quantisation matrix and perform inverse DCT
    dequantised_block = quantized_blocks_Cb(:,:,k) .* jpeg_quant_matrix_chroma;
    reconstructed_block = idct2(dequantised_block);

    % Revert the centring by adding 128
    reconstructed_blocks_Cb(:,:,k) = reconstructed_block + 128;
end

% Reassemble the full subsampled Cb channel
reconstructed_Cb = zeros(rowsC, colsC);
block_idx = 1;
for i = 1:block_size:rowsC
    for j = 1:block_size:colsC  % Step by block_size
        reconstructed_Cb(i:i+block_size-1, j:j+block_size-1) = reconstructed_blocks_Cb(:,:,block_idx);
        block_idx = block_idx + 1;
    end
end

% ------------------------------------
% Reconstruct Chrominance (Cr) Channel
% ------------------------------------
reconstructed_blocks_Cr = zeros(size(blocks_Cr));

for k = 1:num_blocks_Cr
    % Dequantise using the chrominance quantisation matrix and perform inverse DCT
    dequantised_block = quantized_blocks_Cr(:,:,k) .* jpeg_quant_matrix_chroma;
    reconstructed_block = idct2(dequantised_block);

    % Revert the centring by adding 128
    reconstructed_blocks_Cr(:,:,k) = reconstructed_block + 128;
end

% Reassemble the full subsampled Cr channel
reconstructed_Cr = zeros(rowsCr, colsCr);
block_idx = 1;
for i = 1:block_size:rowsCr
    for j = 1:block_size:colsCr  % Step by block_size
        reconstructed_Cr(i:i+block_size-1, j:j+block_size-1) = reconstructed_blocks_Cr(:,:,block_idx);
        block_idx = block_idx + 1;
    end
end

% ---------------------------------
% Upsample the Chrominance Channels
% ---------------------------------

% The Cb and Cr channels were subsampled (4:2:0), so upsampling them to the full resolution
reconstructed_Cb_up = imresize(reconstructed_Cb, [rowsY, colsY], 'bilinear');
reconstructed_Cr_up = imresize(reconstructed_Cr, [rowsY, colsY], 'bilinear');

% -----------------------------------
% Combine Channels and Convert to RGB
% -----------------------------------

% Combine the reconstructed Y, Cb, and Cr channels into a YCbCr image
reconstructed_ycbcr = cat(3, uint8(reconstructed_Y), uint8(reconstructed_Cb_up), uint8(reconstructed_Cr_up));

% Convert the YCbCr image back to RGB for display
reconstructed_rgb = ycbcr2rgb(reconstructed_ycbcr);

% -------------------------------
% Calculate Quality Metrics
% -------------------------------

% Convert the original RGB image to double precision for comparison
original_rgb_double = double(image);
reconstructed_rgb_double = double(reconstructed_rgb);

% Calculate Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)
mse = mean((original_rgb_double - reconstructed_rgb_double).^2, 'all');
psnr_value = 10 * log10((255^2) / mse);

% Compute the absolute difference image and scale for visibility
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

% Overall title with quality metrics
sgtitle(sprintf('Image Reconstruction\nMSE: %.2f, PSNR: %.2f dB', mse, psnr_value));


%% ==== STEP 7.5 Analysing Image Reconstruction at Regions-of-Interest (ROIs) ==== %%
% To inspect the effects of JPEG compression on the overall image, specific
% 75x75 ROIs are extracted from the original RGB image, the reconstructed
% RGB image, and the difference image.

% Define ROI Dimensions
ROI_dim = 75;  

% Each row of ROI_coords defines the [row_start, col_start] for an ROI
ROI_coords = [
    240, 350;  % ROI1
    100, 200;  % ROI2
    400, 380   % ROI3
];
num_ROIs = size(ROI_coords, 1);

% Initialise cell arrays to store ROI slices for each image type
ROI_original = cell(num_ROIs, 1);      
ROI_reconstructed = cell(num_ROIs, 1); 
ROI_difference = cell(num_ROIs, 1);      

% Extract ROIs from the images
for i = 1:num_ROIs
    row_start = ROI_coords(i, 1);
    col_start = ROI_coords(i, 2);
    row_end = row_start + ROI_dim - 1;
    col_end = col_start + ROI_dim - 1;
    
    ROI_original{i} = image(row_start:row_end, col_start:col_end, :);
    ROI_reconstructed{i} = reconstructed_rgb(row_start:row_end, col_start:col_end, :);
    ROI_difference{i} = difference_image_uint8(row_start:row_end, col_start:col_end, :);
end

% Display the ROIs in a 3x3 grid: one row per ROI and three columns:

% Original, Reconstructed, and Difference.
figure('Name', 'Step 7.5: ROI Analysis');

for i = 1:num_ROIs
    subplot(num_ROIs, 3, (i-1)*3+1);
    imshow(ROI_original{i});
    title(sprintf('ROI%d: Original', i));
    
    subplot(num_ROIs, 3, (i-1)*3+2);
    imshow(ROI_reconstructed{i});
    title(sprintf('ROI%d: Reconstructed', i));
    
    subplot(num_ROIs, 3, (i-1)*3+3);
    imshow(ROI_difference{i});
    title(sprintf('ROI%d: Difference', i));
end

sgtitle('Step 7.5: ROI Analysis (Original, Reconstructed, Difference)');

%% ==== STEP 8: Compare Original vs. Compressed Y and Cr Channels ==== %%
% This stage compares the original luminance (Y) and chrominance (Cr) channels
% with their compressed (reconstructed) counterparts. The original Cr channel 
% is displayed in its pseudo‑colour form (Cr_img), and the compressed Cr is derived
% from the reconstructed YCbCr image and converted to a similar pseudo‑colour image.
% The results are presented in a 2x2 grid.

% --- Extract Original Channels ---
original_Y = Y;            % Original luminance (grayscale)
original_Cr_img = Cr_img;  % Original pseudo‑colour Cr channel (from Stage 1)

% --- Compressed Luminance (Y) ---
% 'reconstructed_Y' was reassembled in Stage 7.
compressed_Y = uint8(reconstructed_Y);

% --- Compressed Chrominance (Cr) ---
% Extract the Cr channel from the reconstructed YCbCr image (Stage 7)
compressed_Cr_channel = uint8(reconstructed_ycbcr(:,:,3));

% Use the same neutral value as in Stage 1 to generate a pseudo‑colour representation
neutral = uint8(128 * ones(size(original_Y)));
compressed_Cr_img = ycbcr2rgb(cat(3, neutral, neutral, compressed_Cr_channel));

% --- Display the Results ---
figure('Name', 'Step 8: Comparison of Original vs. Compressed Y and Cr');
subplot(2,2,1);
imshow(original_Y);
title('Original Luminance (Y)');

subplot(2,2,2);
imshow(compressed_Y);
title('Compressed Luminance (Y)');

subplot(2,2,3);
imshow(original_Cr_img);
title('Original Chrominance (Cr)');

subplot(2,2,4);
imshow(compressed_Cr_img);
title('Compressed Chrominance (Cr)');

sgtitle('Stage 8: Comparison of Original vs. Compressed Y and Cr Channels');