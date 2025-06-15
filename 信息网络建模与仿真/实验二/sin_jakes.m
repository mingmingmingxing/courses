% 参数设置
f_m = 200;       % 最大多普勒频移（Hz）
Fs = 100e3;      % 采样率（Hz）
N_samples = 10000; % 样点数
t = (0:N_samples-1)/Fs; % 时间向量

N = 100;         % 振荡器数量

% 生成入射角和随机相位
alpha_n = 2*pi*(0:N-1)/N; % 均匀分布的入射角
phi_n = 2*pi*rand(1, N);  % 随机相位

% 初始化同相和正交分量
I = zeros(1, N_samples);
Q = zeros(1, N_samples);

% 生成I和Q分量
for n = 1:N
    freq = f_m * cos(alpha_n(n)); % 计算多普勒频率
    I = I + sqrt(1/N) * cos(2*pi*freq*t + phi_n(n));
    Q = Q + sqrt(1/N) * sin(2*pi*freq*t + phi_n(n));
end

% 计算自相关函数
[R_I, lags] = xcorr(I, 'biased');
tau = lags / Fs; % 转换为时间延迟

% 理论自相关函数
tau_theory = linspace(-max(tau), max(tau), 1000);
R_theory = besselj(0, 2*pi*f_m * abs(tau_theory));

% 绘制自相关函数
figure;
plot(tau, R_I, 'b');
hold on;
plot(tau_theory, R_theory, 'r--', 'LineWidth', 1.5);
xlabel('时间延迟 \tau (s)');
ylabel('R_I(\tau)');
legend('仿真结果', '理论值');
title('同相分量的自相关函数');
grid on;
xlim([-0.02, 0.02]); % 限制时间延迟范围

% 计算功率谱密度
window = hamming(512);
noverlap = 256;
nfft = 1024;
[Pxx, f] = pwelch(I, window, noverlap, nfft, Fs, 'centered');

% 绘制功率谱密度
figure;
plot(f, 10*log10(Pxx));
xlabel('频率 (Hz)');
ylabel('功率谱密度 (dB/Hz)');
title('同相分量的功率谱密度');
grid on;
xlim([-300, 300]); % 设置频率范围