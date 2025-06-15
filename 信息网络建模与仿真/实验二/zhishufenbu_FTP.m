
function FTP()
    % 生成5000个FTP数据包
    num_ftp_packets = 5000; 
    lambda = 0.5;          % 默认时间间隔参数（平均间隔2秒）
    mean_packet_size = 512; % 默认数据包平均大小
    
    % 生成流量数据
    [ftp_times, ftp_sizes] = generateTraffic(num_ftp_packets, lambda, mean_packet_size);
    
    % 保存为txt文件（时间和大小两列）
    saveToTxt(ftp_times, ftp_sizes, 'ftp_packets.txt');
    
    % 绘制分布图
    plotDistributions(ftp_times, ftp_sizes);
end

% 生成FTP流量数据
function [times, sizes] = generateTraffic(num_packets, lambda, mean_packet_size)
    % 生成时间间隔（指数分布）
    interarrival_times = exprnd(1/lambda, [num_packets, 1]);
    times = cumsum(interarrival_times);
    
    % 生成数据包大小（指数分布）
    sizes = exprnd(mean_packet_size, [num_packets, 1]);
end

% 保存数据到txt文件
function saveToTxt(times, sizes, filename)
    data = [times, sizes];
    dlmwrite(filename, data, 'delimiter', '\t', 'precision', '%.6f');
end

% 绘制时间间隔和数据包大小分布图
function plotDistributions(times, sizes)
    % 时间间隔分布
    interarrival_times = diff([0; times]); % 包含第一个包的时间间隔
    
    figure;
    subplot(2, 1, 1);
    histogram(interarrival_times, 'Normalization', 'pdf');
    title('FTP时间间隔分布（指数分布）');
    xlabel('时间间隔 (秒)');
    ylabel('概率密度');
    
    % 数据包大小分布
    subplot(2, 1, 2);
    histogram(sizes, 'Normalization', 'pdf');
    title('FTP数据包大小分布（指数分布）');
    xlabel('数据包大小 (字节)');
    ylabel('概率密度');
end