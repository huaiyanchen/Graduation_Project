package cn.shaines.filesystem.exception;

/**
 */
public class BusinessException extends RuntimeException {

    public BusinessException(Exception e) {
        super(e);
    }

    public BusinessException(String msg, Exception e) {
        super(msg, e);
    }

    public BusinessException(String msg) {
        super(msg);
    }

}
