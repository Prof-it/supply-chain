function PlotCosts(pop)
Costs=[pop.Cost];
plot(Costs(1,:),Costs(2,:),'ro');
xlabel('1st Objective');
ylabel('2nd Objective');
grid on;
end
