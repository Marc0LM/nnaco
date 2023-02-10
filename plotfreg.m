function plotfreg(X1, YMatrix1)
%CREATEFIGURE(X1, YMATRIX1)
%  X1:  x ���ݵ�ʸ��
%  YMATRIX1:  y ���ݵľ���

%  �� MATLAB �� 16-Mar-2021 22:27:32 �Զ�����

% ���� figure
figure('Color',[1 1 1]);

% ���� axes
axes1 = axes;
hold(axes1,'on');

% ʹ�� plot �ľ������봴������
plot1 = plot(X1,YMatrix1,'Color',[0 0 0]);
set(plot1(1),'DisplayName','ACO-BP','LineStyle',':');
set(plot1(2),'DisplayName','BP');
set(plot1(3),'DisplayName','�Ľ�ACO-BP','LineStyle','--');

% ���� xlabel
xlabel('ѵ������/Epoch','FontSize',10.5);

% ���� ylabel
ylabel('�������','FontSize',10.5);

% ȡ�������е�ע���Ա���������� X ��Χ
% xlim(axes1,[0 60]);
% ȡ�������е�ע���Ա���������� Y ��Χ
% ylim(axes1,[0.6 1.05]);
box(axes1,'on');
% ������������������
set(axes1,'FontName','����');
% ���� legend
legend1 = legend(axes1,'show');
set(legend1,'FontSize',10,'EdgeColor',[1 1 1]);