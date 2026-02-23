# Qwen2 测试计划 - 阶段2历史复盘（Legacy）

> 归档说明：该文档为阶段2期间的问题复盘记录，保留追溯用途，不作为当前执行规则。

1. 问题：流式接口偶发缺少最后 `is_finished=true` 终止块。  
解决：在 `AsyncLLMEngine.stream` 的超时分支补发终止 chunk，并加回归测试。

2. 问题：多线程真实模型并发下出现段错误（析构路径）。  
解决：显式 `close` 链 + C++ `Context` 生命周期修正（初始化、引用语义、跨线程析构风险规避）。

3. 问题：并发首请求 Tokenizer 初始化偶发异常。  
解决：Tokenizer 延迟初始化加锁，Worker 增加编码/解码降级路径。

4. 问题：`test/online/test_online_http.py` 在部分环境报 `http.client.RemoteDisconnected`，且服务端无 traceback。  
根因：测试进程受到环境代理变量影响（`HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`），请求未稳定直连本地 `127.0.0.1` 测试服务。  
解决：在 `test/online/test_online_http.py` 中使用无代理 opener（`urllib.request.ProxyHandler({})`），强制 localhost 直连；README 增加排障说明。

5. 问题：Core 曾存在“伪批处理”路径（`decode` 逐 token 调 `infer`），与动态批处理设计不一致。  
解决：将 `decode` 改为单轮 batch 前向，并在 `KvCache` 引入 unified slot 语义（支持一个 token 关联多个 `seq_id`）。
