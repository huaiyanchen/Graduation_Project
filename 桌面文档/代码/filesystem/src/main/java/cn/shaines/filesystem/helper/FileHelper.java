package cn.shaines.filesystem.helper;

import java.io.IOException;

/**
 * @description 文件存储接口

 */
public interface FileHelper {

    boolean save(String key, byte[] body) throws IOException;

    byte[] findByKey(String key) throws IOException;

    int deleteAllByKeys(String[] keys) throws IOException;

}
