% 添加系统模块路径（示例路径，需替换实际路径）
addpath(fullfile(matlabroot, 'toolbox', 'shared', 'comm_sysmod', 'traffic'));

% 强制刷新路径
rehash toolboxcache;

%% 清空环境
clc;
clear;
close all;

% 设置随机种子，确保结果可重复
rng(1);

% 创建 FTP 流量配置对象
cfgFTP = networkTrafficFTP;
numPkts = 5000;
dt = zeros(1, numPkts);
packetSize = zeros(1, numPkts);

% 生成数据包的到达时间和大小
for packetCount = 1:numPkts
    [dt(packetCount), packetSize(packetCount)] = generate(cfgFTP);
end

% 计算到达时间间隔，第一个数据包的时间间隔设为0
arrivalTimes = cumsum(dt)/1000; % 将时间间隔转换为秒

% 创建矩阵存储数据
dataMatrix = [arrivalTimes', packetSize'];

% 将数据包的到达时间和大小保存到 TXT 文件中
filename = 'FTP_Packet_Data.txt';
writematrix(dataMatrix, filename);

% 可视化包大小
figure;
stem(packetSize); 
title('Packet Size Versus Packet Number');
xlabel('Packet Number');
ylabel('Packet Size in Bytes');

% 可视化时间间隔
figure;
stem(dt); 
title('dt Versus Packet Number');
xlabel('Packet Number');
ylabel('dt in milliseconds');