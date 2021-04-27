git init
git clone https://github.com/bcaitech1/p3-ims-obd-doggum.git

(branch가 존재하지 않으면)
git checkout -b <branch_name> 
(이미 브랜치가 존재하면)
git checkout <branch_name>

------------ 파일 수정 --------------

git add .
git commit -m "<commit_message>"
git push origin <branch_name>

----------- git 저장소로 이동 -----------

compare & pull request 클릭
pr(pull request)에 대한 메시지를 입력하고 create pull request 클릭
팀원끼리 코드리뷰 후 merge 