-- Quantum Task Planner Database Initialization Script
-- PostgreSQL schema for quantum-inspired task planning system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create enum types
CREATE TYPE task_priority AS ENUM (
    'GROUND_STATE',     -- Priority 0 (highest)
    'EXCITED_1',        -- Priority 1
    'EXCITED_2',        -- Priority 2
    'EXCITED_3',        -- Priority 3
    'METASTABLE'        -- Priority 4 (lowest)
);

CREATE TYPE task_state AS ENUM (
    'superposition',
    'entangled',
    'collapsed',
    'completed',
    'failed'
);

CREATE TYPE resource_type AS ENUM (
    'cpu',
    'memory',
    'io',
    'network',
    'storage'
);

-- Tasks table with quantum properties
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    priority task_priority NOT NULL DEFAULT 'EXCITED_2',
    state task_state NOT NULL DEFAULT 'superposition',
    
    -- Quantum properties
    wave_function JSONB DEFAULT '{}',
    entangled_tasks UUID[] DEFAULT '{}',
    coherence_time INTERVAL DEFAULT '30 minutes',
    decoherence_rate FLOAT DEFAULT 0.1,
    
    -- Scheduling properties
    estimated_duration INTERVAL,
    actual_duration INTERVAL,
    scheduled_start TIMESTAMPTZ,
    scheduled_end TIMESTAMPTZ,
    actual_start TIMESTAMPTZ,
    actual_end TIMESTAMPTZ,
    deadline TIMESTAMPTZ,
    
    -- Resource requirements
    resources_required JSONB DEFAULT '{}',
    
    -- Dependencies
    dependencies UUID[] DEFAULT '{}',
    blocks UUID[] DEFAULT '{}',
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(255),
    
    -- Search and indexing
    search_vector tsvector
);

-- Quantum resources table
CREATE TABLE quantum_resources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    resource_type resource_type NOT NULL,
    
    -- Capacity management
    total_capacity FLOAT NOT NULL,
    available_capacity FLOAT NOT NULL,
    reserved_capacity FLOAT DEFAULT 0,
    
    -- Quantum properties
    quantum_efficiency FLOAT DEFAULT 1.0,
    coherence_time INTERVAL DEFAULT '1 hour',
    error_rate FLOAT DEFAULT 0.01,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_calibration TIMESTAMPTZ DEFAULT NOW(),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_capacity CHECK (available_capacity >= 0),
    CONSTRAINT valid_total_capacity CHECK (total_capacity > 0),
    CONSTRAINT valid_quantum_efficiency CHECK (quantum_efficiency > 0)
);

-- Task execution history
CREATE TABLE task_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    
    -- Execution details
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status VARCHAR(50) NOT NULL,
    exit_code INTEGER,
    
    -- Resource utilization
    resources_used JSONB DEFAULT '{}',
    quantum_speedup FLOAT,
    
    -- Output and logs
    stdout TEXT,
    stderr TEXT,
    logs JSONB DEFAULT '{}',
    
    -- Quantum measurements
    final_state task_state,
    measurement_results JSONB DEFAULT '{}',
    entanglement_broken BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Quantum entanglements tracking
CREATE TABLE task_entanglements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_a_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    task_b_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    
    -- Entanglement properties
    entanglement_strength FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    broken_at TIMESTAMPTZ,
    
    -- Bell state correlation
    correlation_coefficient FLOAT,
    
    CONSTRAINT different_tasks CHECK (task_a_id != task_b_id),
    CONSTRAINT unique_entanglement UNIQUE (task_a_id, task_b_id)
);

-- Resource reservations
CREATE TABLE resource_reservations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id UUID REFERENCES quantum_resources(id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    
    -- Reservation details
    reserved_amount FLOAT NOT NULL,
    reserved_at TIMESTAMPTZ DEFAULT NOW(),
    released_at TIMESTAMPTZ,
    
    -- Quantum scheduling
    priority_boost FLOAT DEFAULT 0,
    quantum_advantage BOOLEAN DEFAULT FALSE,
    
    CONSTRAINT valid_reservation CHECK (reserved_amount > 0)
);

-- Quantum measurements and metrics
CREATE TABLE quantum_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(50),
    
    -- Context
    task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    resource_id UUID REFERENCES quantum_resources(id) ON DELETE SET NULL,
    
    -- Quantum properties
    uncertainty FLOAT DEFAULT 0,
    coherence_time FLOAT,
    measurement_basis VARCHAR(100),
    
    -- Timestamp
    measured_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Metadata
    tags JSONB DEFAULT '{}'
);

-- Quantum optimization plans
CREATE TABLE optimization_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_name VARCHAR(255) NOT NULL,
    
    -- Plan configuration
    max_iterations INTEGER DEFAULT 1000,
    convergence_threshold FLOAT DEFAULT 0.001,
    temperature_schedule JSONB DEFAULT '{}',
    
    -- Results
    final_schedule JSONB DEFAULT '{}',
    optimization_score FLOAT,
    iterations_completed INTEGER,
    convergence_achieved BOOLEAN DEFAULT FALSE,
    
    -- Quantum annealing parameters
    annealing_time INTERVAL,
    quantum_fluctuations FLOAT DEFAULT 0.1,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    created_by VARCHAR(255)
);

-- Indexes for performance
CREATE INDEX idx_tasks_task_id ON tasks(task_id);
CREATE INDEX idx_tasks_priority ON tasks(priority);
CREATE INDEX idx_tasks_state ON tasks(state);
CREATE INDEX idx_tasks_scheduled_start ON tasks(scheduled_start);
CREATE INDEX idx_tasks_deadline ON tasks(deadline);
CREATE INDEX idx_tasks_created_at ON tasks(created_at);
CREATE INDEX idx_tasks_search_vector ON tasks USING gin(search_vector);
CREATE INDEX idx_tasks_dependencies ON tasks USING gin(dependencies);
CREATE INDEX idx_tasks_entangled_tasks ON tasks USING gin(entangled_tasks);

CREATE INDEX idx_resources_name ON quantum_resources(name);
CREATE INDEX idx_resources_type ON quantum_resources(resource_type);
CREATE INDEX idx_resources_active ON quantum_resources(is_active);

CREATE INDEX idx_executions_task_id ON task_executions(task_id);
CREATE INDEX idx_executions_started_at ON task_executions(started_at);
CREATE INDEX idx_executions_status ON task_executions(status);

CREATE INDEX idx_entanglements_task_a ON task_entanglements(task_a_id);
CREATE INDEX idx_entanglements_task_b ON task_entanglements(task_b_id);
CREATE INDEX idx_entanglements_created_at ON task_entanglements(created_at);

CREATE INDEX idx_reservations_resource_id ON resource_reservations(resource_id);
CREATE INDEX idx_reservations_task_id ON resource_reservations(task_id);
CREATE INDEX idx_reservations_reserved_at ON resource_reservations(reserved_at);

CREATE INDEX idx_metrics_name ON quantum_metrics(metric_name);
CREATE INDEX idx_metrics_measured_at ON quantum_metrics(measured_at);
CREATE INDEX idx_metrics_task_id ON quantum_metrics(task_id);

-- Triggers for automatic updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_tasks_updated_at 
    BEFORE UPDATE ON tasks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_resources_updated_at 
    BEFORE UPDATE ON quantum_resources 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update search vector
CREATE OR REPLACE FUNCTION update_task_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.task_id, '')), 'C');
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_task_search_vector_trigger
    BEFORE INSERT OR UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_task_search_vector();

-- Function to manage resource capacity
CREATE OR REPLACE FUNCTION check_resource_capacity()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if reservation would exceed capacity
    IF (SELECT available_capacity FROM quantum_resources WHERE id = NEW.resource_id) < NEW.reserved_amount THEN
        RAISE EXCEPTION 'Insufficient resource capacity for reservation';
    END IF;
    
    -- Update available capacity
    UPDATE quantum_resources 
    SET available_capacity = available_capacity - NEW.reserved_amount,
        reserved_capacity = reserved_capacity + NEW.reserved_amount
    WHERE id = NEW.resource_id;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER check_resource_capacity_trigger
    BEFORE INSERT ON resource_reservations
    FOR EACH ROW EXECUTE FUNCTION check_resource_capacity();

-- Function to release resources when reservation is deleted
CREATE OR REPLACE FUNCTION release_resource_capacity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE quantum_resources 
    SET available_capacity = available_capacity + OLD.reserved_amount,
        reserved_capacity = reserved_capacity - OLD.reserved_amount
    WHERE id = OLD.resource_id;
    
    RETURN OLD;
END;
$$ language 'plpgsql';

CREATE TRIGGER release_resource_capacity_trigger
    AFTER DELETE ON resource_reservations
    FOR EACH ROW EXECUTE FUNCTION release_resource_capacity();

-- Initial data: Create default quantum resources
INSERT INTO quantum_resources (name, resource_type, total_capacity, available_capacity, quantum_efficiency) VALUES
('quantum_cpu_1', 'cpu', 4.0, 4.0, 2.0),
('quantum_cpu_2', 'cpu', 4.0, 4.0, 1.8),
('quantum_memory', 'memory', 16.0, 16.0, 1.5),
('quantum_io', 'io', 8.0, 8.0, 1.2);

-- Views for common queries
CREATE OR REPLACE VIEW active_tasks AS
SELECT 
    t.*,
    EXTRACT(EPOCH FROM (NOW() - t.created_at))/3600 as age_hours,
    CASE 
        WHEN t.deadline IS NOT NULL THEN EXTRACT(EPOCH FROM (t.deadline - NOW()))/3600
        ELSE NULL 
    END as hours_until_deadline
FROM tasks t
WHERE t.state NOT IN ('completed', 'failed');

CREATE OR REPLACE VIEW resource_utilization AS
SELECT 
    r.name,
    r.resource_type,
    r.total_capacity,
    r.available_capacity,
    r.reserved_capacity,
    ROUND((r.reserved_capacity / r.total_capacity * 100)::numeric, 2) as utilization_percentage,
    r.quantum_efficiency,
    r.is_active
FROM quantum_resources r;

CREATE OR REPLACE VIEW task_dependencies_view AS
SELECT 
    t1.task_id as task,
    t1.name as task_name,
    t2.task_id as depends_on,
    t2.name as depends_on_name,
    t2.state as dependency_state
FROM tasks t1
CROSS JOIN LATERAL unnest(t1.dependencies) as dep_id
JOIN tasks t2 ON t2.id = dep_id;

-- Performance monitoring view
CREATE OR REPLACE VIEW quantum_performance_metrics AS
SELECT 
    DATE_TRUNC('hour', te.started_at) as hour,
    COUNT(*) as tasks_executed,
    AVG(EXTRACT(EPOCH FROM (te.completed_at - te.started_at))) as avg_execution_time_seconds,
    AVG(te.quantum_speedup) as avg_quantum_speedup,
    COUNT(*) FILTER (WHERE te.status = 'completed') as successful_tasks,
    COUNT(*) FILTER (WHERE te.status = 'failed') as failed_tasks
FROM task_executions te
WHERE te.started_at >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', te.started_at)
ORDER BY hour DESC;

-- Grant permissions (adjust as needed for your deployment)
-- Note: In production, create specific users with limited permissions
CREATE USER quantum_api WITH ENCRYPTED PASSWORD 'quantum_secure_password_change_in_production';
GRANT CONNECT ON DATABASE quantum_planner TO quantum_api;
GRANT USAGE ON SCHEMA public TO quantum_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO quantum_api;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO quantum_api;
GRANT SELECT ON ALL VIEWS IN SCHEMA public TO quantum_api;

-- Enable row level security (optional, for multi-tenant deployments)
-- ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE task_executions ENABLE ROW LEVEL SECURITY;

COMMIT;