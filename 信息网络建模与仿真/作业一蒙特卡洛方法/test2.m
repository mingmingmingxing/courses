% 二进制反极性基带通信系统蒙特卡洛仿真
clc;
clear;
close all;

% 参数设置
Eb = 1; % 信号能量
EbN0_dB = -2:2:8; % 信噪比范围(dB)
EbN0 = 10.^(EbN0_dB/10); % 线性信噪比
N0 = Eb ./ EbN0; % 噪声功率谱密度

% 方法1: 固定传输比特数(10000)
num_bits_fixed = 10000;
num_experiments = 50;
ber_fixed = zeros(length(EbN0_dB), num_experiments);

% 方法2: 固定错误比特数(100)
min_errors = 100;
ber_variable = zeros(length(EbN0_dB), num_experiments);
num_bits_used = zeros(length(EbN0_dB), num_experiments);

% 开始仿真
for exp_idx = 1:num_experiments
    for snr_idx = 1:length(EbN0_dB)
        % 方法1: 固定比特数
        % 生成随机比特序列
        bits = randi([0 1], 1, num_bits_fixed);
        % 调制: 0->+sqrt(Eb), 1->-sqrt(Eb)
        modulated = (1 - 2*bits) * sqrt(Eb);
        % 计算噪声标准差
        sigma = sqrt(N0(snr_idx)/2);
        % 添加高斯噪声
        received = modulated + sigma * randn(1, num_bits_fixed);
        % 判决
        decoded = (received < 0);
        % 计算误码数
        errors = sum(bits ~= decoded);
        % 存储BER
        ber_fixed(snr_idx, exp_idx) = errors / num_bits_fixed;
        
        % 方法2: 固定错误数
        errors = 0;
        total_bits = 0;
        while errors < min_errors
            % 每次生成1000比特以减少循环次数
            batch_size = 1000;
            bits = randi([0 1], 1, batch_size);
            modulated = (1 - 2*bits) * sqrt(Eb);
            received = modulated + sigma * randn(1, batch_size);
            decoded = (received < 0);
            new_errors = sum(bits ~= decoded);
            
            errors = errors + new_errors;
            total_bits = total_bits + batch_size;
        end
        % 存储BER和使用的比特数
        ber_variable(snr_idx, exp_idx) = errors / total_bits;
        num_bits_used(snr_idx, exp_idx) = total_bits;
    end
end

% 理论误码率
theory_ber = 0.5 * erfc(sqrt(EbN0));

% 绘图
figure;

% 方法1结果: 固定比特数
subplot(2,1,1);
semilogy(EbN0_dB, ber_fixed, 'o-');
hold on;
semilogy(EbN0_dB, theory_ber, 'k-', 'LineWidth', 2);
title('固定传输比特数(10000 bits)');
xlabel('Eb/N0 (dB)');
ylabel('BER');
grid on;
legend('实验曲线', '理论曲线', 'Location', 'southwest');
ylim([1e-5 1]);

% 方法2结果: 固定错误数
subplot(2,1,2);
semilogy(EbN0_dB, ber_variable, 'o-');
hold on;
semilogy(EbN0_dB, theory_ber, 'k-', 'LineWidth', 2);
title('固定错误比特数(100 errors)');
xlabel('Eb/N0 (dB)');
ylabel('BER');
grid on;
legend('实验曲线', '理论曲线', 'Location', 'southwest');
ylim([1e-5 1]);

% 绘制使用的比特数
figure;
semilogy(EbN0_dB, mean(num_bits_used, 2), 'o-');
title('不同Eb/N0下需要的平均比特数(固定100个错误)');
xlabel('Eb/N0 (dB)');
ylabel('平均传输比特数');
grid on;