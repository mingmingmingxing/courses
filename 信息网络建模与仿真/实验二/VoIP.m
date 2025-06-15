% 添加系统模块路径（示例路径，需替换实际路径）
addpath(fullfile(matlabroot, 'toolbox', 'shared', 'comm_sysmod', 'traffic'));

% 强制刷新路径
rehash toolboxcache;

%% 清空环境
clc;
clear;
close all;

% 创建 networkTrafficVoIP 对象
voipObj = networkTrafficVoIP;

% 初始化变量以存储时间和数据包大小
numPackets = 400;
t = zeros(1, numPackets);
sizes = zeros(1, numPackets);

% 循环生成 400 个数据包
for i = 1:numPackets
    [dt, packetSize] = generate(voipObj);
    if i == 1
        t(i) = dt;
    else
        t(i) = t(i-1) + dt;
    end
    sizes(i) = packetSize;
end

% 将数据包产生的时间和大小存储在文件中
data = [t', sizes'];
filename = 'voip_packets.txt';
dlmwrite(filename, data, 'delimiter', '\t', 'precision', 6);

% 计算数据包时间间隔
timeIntervals = diff(t);

% 画出数据包时间间隔的分布图
figure;
histogram(timeIntervals, 'Normalization', 'pdf');
title('VoIP 数据包时间间隔分布图');
xlabel('时间间隔 (秒)');
ylabel('概率密度');    