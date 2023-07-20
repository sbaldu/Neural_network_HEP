#

values=(100 200 300 400 500 600)
  
make 

for x in ${!values[@]}
do
  value=${values[$x]}
  file_name="values_${value}.txt"
  touch $file_name
  ./serial.out $value > $file_name
done
