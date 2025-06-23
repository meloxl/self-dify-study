### 5.dify综合应用
面试宝典

报错：429调用超频了。
修改.env文件，设置CELERY_WORKER_AMOUNT=1  # 原默认值可能为 4-8，调整为 1 以降低并发。


![面试宝典.png](images/面试宝典.png)