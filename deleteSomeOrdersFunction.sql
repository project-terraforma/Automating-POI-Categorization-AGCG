CREATE OR REPLACE FUNCTION deleteSomeOrdersFunction(maxOrderDeletions INT)
RETURNS INTEGER AS $$
DECLARE
    total_deleted INT := 0;
    supplier_data RECORD;
    deleted_count INT;
BEGIN
    -- Input validation: Reject invalid input
    IF maxOrderDeletions <= 0 THEN
        RETURN -1;
    END IF;

    -- Loop over suppliers sorted by cancellation count and name
    FOR supplier_data IN
        WITH SupplierStats AS (
            SELECT
                s.supplierID,
                
                -- Count of FUTURE orders (to potentially delete)
                COUNT(*) FILTER (
                    WHERE o.orderDate > '2024-01-05'
                ) AS future_orders,
                
                -- Count of CANCELLED past orders
                COUNT(*) FILTER (
                    WHERE o.orderDate <= '2024-01-05' AND o.status = 'cncl'
                ) AS canceled_past_orders

            FROM Supplier s
            JOIN OrderSupply o ON s.supplierID = o.supplierID
            GROUP BY s.supplierID
        )
        SELECT *
        FROM SupplierStats
        WHERE future_orders > 0
        ORDER BY canceled_past_orders DESC, supplierID ASC
    LOOP
        -- If deleting this supplier’s future orders won’t exceed the max limit
        IF (total_deleted + supplier_data.future_orders) <= maxOrderDeletions THEN
            -- Delete the supplier’s future orders
            DELETE FROM OrderSupply
            WHERE supplierID = supplier_data.supplierID
              AND orderDate > '2024-01-05';

            -- Get how many rows were deleted in this DELETE
            GET DIAGNOSTICS deleted_count = ROW_COUNT;

            -- Update our running total
            total_deleted := total_deleted + deleted_count;
        ELSE
            -- Can’t fit next supplier’s orders — return current total
            RETURN total_deleted;
        END IF;
    END LOOP;

    -- If loop completes without breaking early, return the full count
    RETURN total_deleted;
END;
$$ LANGUAGE plpgsql;
