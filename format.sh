#

files=$(find . \( -name "*.cc" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" \))

for file in $files
do
  clang-format -style=file -i $file
done
