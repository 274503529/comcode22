clear all
clc
close all
load YA.mat;
load rstrength.mat;
data=YA;
[M,N]=size(data);
strength=strength;
attributes=ones(M,N-1);
for i=1:N-1
    attributes(:,i)=data(:,i);
end
P_train = attributes(:,1:226);
T_train = strength(1:226)';  

P_test = attributes(:,227:281);
T_test = strength(227:281)';
value=0.95;
i=1;
  for i=1:20
  for j=1:20
net=newff(P_train,T_train,[i j]);
view(net)  
w1=net.iw{1,1};
theta1=net.b{1};
w2=net.lw{2,1};  
theta2=net.b{2}; 
 net.trainParam.epochs=2000; 
 net.trainParam.goal=1e-3; 
 net.trainParam.lr=0.01; 

net=train(net,P_train,T_train);
t_sim=sim(net,P_test); 
p_sim=sim(net,P_train); 
T_sim=t_sim;
P_sim=p_sim;
Predict_1=P_sim;
Predict_2=T_sim; 
result_1 = [T_train' Predict_1'];
result_2 = [T_test' Predict_2']; 
[r,p]=corr(T_test',Predict_2'); 
[Pr,Pp]=corr(T_test',Predict_2','type','Pearson');
A=T_test';
B=Predict_2';
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
  if Pr>value
    break
end
 i=i+1
end 
test_error = A - B;
MSE =  mean(test_error.^2);
RMSE1=sqrt(MSE);
NN=NSE(B, A) 
j=j+1
end
close all