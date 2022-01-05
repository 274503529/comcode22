clear all
clc
load YA.mat;
load strength.mat;
data=YA;
[M,N]=size(data);
strength=strength;
attributes=ones(M,N-1);
for i=1:N-1
    attributes(:,i)=data(:,i);
end
p_train = attributes(:,1:226)';
t_train = strength(1:226);  
p_test = attributes(:,227:281)';
t_test = strength(227:281);

pn_train = p_train;
pn_test = p_test;

tn_train = t_train;
tn_test = t_test;
i=1;
[c,g] = meshgrid(-12:0.2:12,-12:0.2:12);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 0;
bestg = 0;
error = Inf;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];
        cg(i,j) = svmtrain(tn_train,pn_train,cmd);
        if cg(i,j) < error
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
        if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
    end
end
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01']; 
 model = svmtrain(tn_train,pn_train,cmd);
[Predict_1,error_1] = svmpredict(tn_train,pn_train,model);
[Predict_2,error_2] = svmpredict(tn_test,pn_test,model);
predict_1 = Predict_1;
predict_2 = Predict_2;
result_1 = [t_train predict_1];
result_2 = [t_test predict_2]; 
[r,p]=corr(t_test,predict_2); 
[Pr,Pp]=corr(t_test,predict_2,'type','Pearson');
figure(1)
plot(1:length(t_train),t_train,'r-*',1:length(t_train),predict_1,'b:o')
grid on
legend('T','F')
xlabel('GWL')
ylabel('gwl')
string_1 = {'com';
           ['mse = ' num2str(error_1(2)) ' R^2 = ' num2str(error_1(3))]};
title(string_1)
figure(2)
plot(1:length(t_test),t_test,'r-*',1:length(t_test),predict_2,'b:o')
grid on
legend('T','F')
xlabel('GWL')
ylabel('gwl')
string_2 = {'com';
           ['mse = ' num2str(error_2(2)) ' R^2 = ' num2str(error_2(3))]};
title(string_2)
A=t_test;
B=predict_2;
[r,p]=corr(A,B);
[Pr,Pp]=corr(A,B,'type','Pearson');
[Sr,Sp]=corr(A,B,'type','Spearman'); 
AA = A';
BB= B';
n=size(AA);
m=n(2);
std((BB-AA),1) ;                          
RMSE=sqrt(sum((BB-AA).^2)/m) ; 
MAE=mean(abs(AA-BB));
test_error = A - B;
MSE =  mean(test_error.^2);
RMSE1=sqrt(MSE);
N=NSE(B, A) 




