function flicker_noise_demo()
    % 参数设置
    fs = 1000;          % 采样频率 (Hz)
    T = 10;             % 信号时长 (秒)
    N = fs * T;         % 采样点数
    f0 = 0.1;           % 截止频率 (Hz)
    filter_order = 20;  % 滤波器阶数
    
    % 生成独立高斯白噪声
    white_noise = randn(N, 1);
    
    % 设计IIR滤波器
    [b, a] = design_flicker_filter(fs, f0, filter_order);
    
    % 滤波生成闪烁噪声
    flicker_noise = filter(b, a, white_noise);
    
    % 归一化
    flicker_noise = flicker_noise / std(flicker_noise);
    
    % 绘制时域信号
    figure;
    subplot(2,1,1);
    t = (0:N-1)/fs;
    plot(t, flicker_noise);
    xlabel('时间 (s)');
    ylabel('幅值');
    title('闪烁噪声时域信号');
    grid on;
    
    % 计算并绘制功率谱密度
    subplot(2,1,2);
    [pxx, f] = pwelch(flicker_noise, hanning(N/8), [], [], fs);
    loglog(f, pxx, 'b');
    hold on;
    loglog(f, 2./f, 'r--', 'LineWidth', 2);
    xlabel('频率 (Hz)');
    ylabel('功率谱密度');
    title('功率谱密度比较');
    legend('生成的噪声', '理论S(f)=2/f');
    grid on;
    
    % 计算自相关函数
    figure;
    max_lag = 1000;
    [acf, lags] = xcorr(flicker_noise, max_lag, 'coeff');
    lags = lags(max_lag+1:end)/fs;
    acf = acf(max_lag+1:end);
    plot(lags, acf);
    xlabel('时延 (s)');
    ylabel('自相关函数');
    title('闪烁噪声自相关函数');
    grid on;
end

function [b, a] = design_flicker_filter(fs, f0, order)
    % 设计用于生成闪烁噪声的IIR滤波器
    
    % 频率点 (归一化到[0,1], 1对应Nyquist频率)
    f = linspace(0, 1, 1000);
    
    % 期望的幅度响应
    desired_mag = zeros(size(f));
    for i = 1:length(f)
        freq = f(i) * fs/2; % 转换为实际频率
        if abs(freq) >= f0
            desired_mag(i) = 1/sqrt(abs(freq));
        else
            desired_mag(i) = 0;
        end
    end
    
    % 归一化
    desired_mag = desired_mag / max(desired_mag);
    
    % 使用yulewalk设计滤波器
    [b, a] = yulewalk(order, f, desired_mag);
end