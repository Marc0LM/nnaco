function plotfreg(X1, YMatrix1)
%CREATEFIGURE(X1, YMATRIX1)
%  X1:  x 数据的矢量
%  YMATRIX1:  y 数据的矩阵

%  由 MATLAB 于 16-Mar-2021 22:27:32 自动生成

% 创建 figure
figure('Color',[1 1 1]);

% 创建 axes
axes1 = axes;
hold(axes1,'on');

% 使用 plot 的矩阵输入创建多行
plot1 = plot(X1,YMatrix1,'Color',[0 0 0]);
set(plot1(1),'DisplayName','ACO-BP','LineStyle',':');
set(plot1(2),'DisplayName','BP');
set(plot1(3),'DisplayName','改进ACO-BP','LineStyle','--');

% 创建 xlabel
xlabel('训练代数/Epoch','FontSize',10.5);

% 创建 ylabel
ylabel('均方误差','FontSize',10.5);

% 取消以下行的注释以保留坐标轴的 X 范围
% xlim(axes1,[0 60]);
% 取消以下行的注释以保留坐标轴的 Y 范围
% ylim(axes1,[0.6 1.05]);
box(axes1,'on');
% 设置其余坐标轴属性
set(axes1,'FontName','宋体');
% 创建 legend
legend1 = legend(axes1,'show');
set(legend1,'FontSize',10,'EdgeColor',[1 1 1]);