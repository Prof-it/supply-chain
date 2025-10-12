figure(2);
hold on
Costs=[];
for i=2:length(F)
    F2=pop(F{i});
    Costs=[Costs , F2.Cost];
end
plot(Costs(1,:),Costs(2,:),'ro');
xlabel('1st Objective');
ylabel('2nd Objective');
grid on;
F1=pop(F{1});
Costs=[F1.Cost];
plot(Costs(1,:),Costs(2,:),'bo');
legend('Other-Frontiers','First-Frontiers')
