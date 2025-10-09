
#!/bin/bash

# ouromamba_dataset 디렉토리에서 실행
cd ./dataset/ouromamba_dataset

for dir in train val test; do
    if [ -d "$dir" ]; then
        echo "Processing $dir directory..."
        cd "$dir"
        
        # single_class 디렉토리 생성
        mkdir -p single_class
        
        # 모든 class_* 디렉토리에서 이미지를 single_class로 이동
        for class_dir in class_*; do
            if [ -d "$class_dir" ]; then
                echo "Moving images from $class_dir to single_class..."
                mv "$class_dir"/* single_class/ 2>/dev/null
                rmdir "$class_dir"
            fi
        done
        
        # single_class 디렉토리의 파일 개수 확인
        img_count=$(ls single_class/*.png 2>/dev/null | wc -l)
        echo "Total images in $dir/single_class: $img_count"
        
        cd ..
    fi
done

echo "Dataset reorganization complete!"
echo "Now you have:"
echo "- train/single_class/ with all training images"
echo "- val/single_class/ with all validation images" 
echo "- test/single_class/ with all test images"