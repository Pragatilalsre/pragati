use Sample_database

create table Employee(
  Emp_Id int primary key,
  Emp_name Varchar(255),
  City Varchar(255),
  Salary Varchar(255),
  Joining_date date ,
  Dept_id int ,
  Manager_id int
);
create table Department (
  Dept_id int primary key ,
  Dept_Name Varchar(255)
);

Create table Hierarchy(
  Manager_id int primary key,
  Manager_name Varchar(255),
  Emp_id int
)

ALTER TABLE Employee
ADD FOREIGN KEY (Dept_id) REFERENCES Department(Dept_id) ;
ALTER TABLE Employee
ADD FOREIGN KEY (Manager_id) REFERENCES Hierarchy(Manager_id) ;


select * from Employee;

select * from Employee inner join Department on Employee.Dept_id = Department.Dept_id;
select * from Employee inner join hierarchy on hierarchy.Manager_id = Employee.Manager_id;

CREATE PROCEDURE Manager_Name( Emp_id )
BEGIN
    SELECT E.Emp_id, E.Emp_name , E.Manager_id , H.Manager_name  FROM Employee E inner join Hierarchy H on H.Manager_id = E.Manager_id where E.Emp_id = 1

END;

call Manager_name(5);



CREATE PROCEDURE Employee_Detail_list(IN date1 Date , IN date2 Date)
BEGIN
    SELECT E.emp_name , E.city, E.salary , D.dept_nme, E.joining_date   FROM Employee E  inner join Department D on E.Dept_id = D.Dept_id
    where E.joining_date between date1 and date2;

END;

Call Employee_detail_list()